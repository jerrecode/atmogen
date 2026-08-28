from __future__ import annotations

"""Stiff, element-conserving local chemical-kinetics integration.

The solver operates on molar concentrations [mol m^-3]. Thermal reactions use an
Arrhenius law and photolysis reactions consume externally supplied local J rates
[s^-1]. The module intentionally ships no fabricated planetary reaction network;
scientific networks are data and must carry their own provenance/validity metadata.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from .database import (BUILTIN_DATABASE, ChemicalDatabase, Reaction,
                       canonical_species, reaction_element_residual)

R_GAS = 8.31446261815324


@dataclass(frozen=True, slots=True)
class KineticsResult:
    time_s: np.ndarray
    concentration_mol_m3: Mapping[str, np.ndarray]
    final_concentration_mol_m3: Mapping[str, float]
    reaction_rates_mol_m3_s: Mapping[str, float]
    converged: bool
    method: str
    nfev: int
    njev: int
    nlu: int
    minimum_raw_concentration_mol_m3: float
    element_relative_residual: float
    message: str


def arrhenius_rate_constant(reaction: Reaction, temperature_k: float) -> float:
    """Evaluate a reaction rate constant using the reaction's SI parameterization."""
    if reaction.rate_law != "arrhenius":
        raise ValueError("arrhenius_rate_constant requires an arrhenius reaction")
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_k must be finite and positive")
    assert reaction.pre_exponential_factor_si is not None
    exponent = -reaction.activation_energy_j_mol / (R_GAS * temperature)
    exponent = float(np.clip(exponent, -745.0, 700.0))
    value = reaction.pre_exponential_factor_si
    value *= (temperature / reaction.reference_temperature_k) ** reaction.temperature_exponent
    value *= np.exp(exponent)
    if not np.isfinite(value) or value < 0:
        raise FloatingPointError(f"invalid rate constant for reaction {reaction.key!r}: {value!r}")
    return float(value)


def expected_rate_coefficient_units(reaction_order: float) -> str:
    order = float(reaction_order)
    if not np.isfinite(order) or order <= 0:
        raise ValueError("reaction_order must be finite and positive")
    if abs(order - 1.0) <= 1e-12:
        return "s^-1"
    return f"(m^3 mol^-1)^{order - 1:g} s^-1"


def _canonical_reaction(reaction: Reaction, database: ChemicalDatabase) -> Reaction:
    reactants = {canonical_species(key): float(value) for key, value in reaction.reactants.items()}
    products = {canonical_species(key): float(value) for key, value in reaction.products.items()}
    for key in set(reactants) | set(products):
        database.get(key)
    canonical = Reaction(
        key=reaction.key,
        reactants=reactants,
        products=products,
        rate_law=reaction.rate_law,
        pre_exponential_factor_si=reaction.pre_exponential_factor_si,
        temperature_exponent=reaction.temperature_exponent,
        activation_energy_j_mol=reaction.activation_energy_j_mol,
        reference_temperature_k=reaction.reference_temperature_k,
        rate_coefficient_units=reaction.rate_coefficient_units,
        provenance_class=reaction.provenance_class,
        source=reaction.source,
        validity=reaction.validity,
    )
    residual = reaction_element_residual(canonical, database.species)
    if residual and max(abs(value) for value in residual.values()) > 1e-10:
        raise ValueError(f"reaction {reaction.key!r} is not element-balanced: {residual}")
    return canonical


def _element_totals(concentrations: Mapping[str, float], database: ChemicalDatabase) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_key, concentration in concentrations.items():
        amount = float(concentration)
        if amount == 0:
            continue
        species = database.get(raw_key)
        for element, count in species.formula.items():
            totals[element] = totals.get(element, 0.0) + amount * float(count)
    return totals


def _relative_element_residual(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    elements = set(before) | set(after)
    if not elements:
        return 0.0
    return float(max(
        abs(float(after.get(element, 0.0)) - float(before.get(element, 0.0)))
        / max(abs(float(before.get(element, 0.0))), 1.0e-30)
        for element in elements
    ))


def _prepare_network(
    reactions: Sequence[Reaction],
    initial_concentration_mol_m3: Mapping[str, float],
    database: ChemicalDatabase,
) -> tuple[tuple[Reaction, ...], tuple[str, ...], np.ndarray, csr_matrix]:
    if not reactions:
        raise ValueError("kinetics network cannot be empty")
    canonical: list[Reaction] = []
    keys: set[str] = set()
    seen: set[str] = set()
    for reaction in reactions:
        item = _canonical_reaction(reaction, database)
        if item.key in seen:
            raise ValueError(f"duplicate reaction key {item.key!r}")
        seen.add(item.key)
        canonical.append(item)
        keys.update(item.reactants)
        keys.update(item.products)
    for raw_key, value in initial_concentration_mol_m3.items():
        key = canonical_species(raw_key)
        database.get(key)
        concentration = float(value)
        if not np.isfinite(concentration) or concentration < 0:
            raise ValueError(f"invalid initial concentration {raw_key!r}: {value!r}")
        keys.add(key)
    species = tuple(sorted(keys))
    index = {key: i for i, key in enumerate(species)}
    stoich = np.zeros((len(species), len(canonical)), dtype=float)
    dependency = np.zeros((len(species), len(species)), dtype=bool)
    for reaction_index, reaction in enumerate(canonical):
        affected: set[str] = set()
        for key, coefficient in reaction.reactants.items():
            stoich[index[key], reaction_index] -= float(coefficient)
            affected.add(key)
        for key, coefficient in reaction.products.items():
            stoich[index[key], reaction_index] += float(coefficient)
            affected.add(key)
        for output_key in affected:
            for reactant_key in reaction.reactants:
                dependency[index[output_key], index[reactant_key]] = True
    return tuple(canonical), species, stoich, csr_matrix(dependency)


def reaction_rates(
    *,
    reactions: Sequence[Reaction],
    concentration_mol_m3: Mapping[str, float],
    temperature_k: float,
    photolysis_rates_s1: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate local volumetric reaction rates [mol m^-3 s^-1]."""
    photolysis = photolysis_rates_s1 or {}
    rates: dict[str, float] = {}
    for reaction in reactions:
        if reaction.rate_law == "arrhenius":
            k = arrhenius_rate_constant(reaction, temperature_k)
            rate = k
            for key, coefficient in reaction.reactants.items():
                c = max(float(concentration_mol_m3.get(key, 0.0)), 0.0)
                rate *= c ** float(coefficient)
        else:
            if reaction.key not in photolysis:
                raise KeyError(f"missing photolysis rate J for reaction {reaction.key!r}")
            j_rate = float(photolysis[reaction.key])
            if not np.isfinite(j_rate) or j_rate < 0:
                raise ValueError(f"invalid photolysis rate for {reaction.key!r}: {j_rate!r}")
            parent = next(iter(reaction.reactants))
            rate = j_rate * max(float(concentration_mol_m3.get(parent, 0.0)), 0.0)
        if not np.isfinite(rate) or rate < 0:
            raise FloatingPointError(f"invalid volumetric rate for reaction {reaction.key!r}: {rate!r}")
        rates[reaction.key] = float(rate)
    return rates


def integrate_kinetics(
    *,
    reactions: Sequence[Reaction],
    initial_concentration_mol_m3: Mapping[str, float],
    temperature_k: float,
    duration_s: float,
    database: ChemicalDatabase = BUILTIN_DATABASE,
    photolysis_rates_s1: Mapping[str, float] | None = None,
    method: str = "BDF",
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance_mol_m3: float = 1.0e-12,
    sample_times_s: np.ndarray | None = None,
) -> KineticsResult:
    """Integrate a closed local reaction network using a stiff implicit solver.

    ``BDF`` and ``Radau`` are supported intentionally because atmospheric reaction
    networks are commonly stiff. Solver success is not trusted alone: the result is
    also checked for significant negative concentrations and elemental closure.
    """
    temperature = float(temperature_k)
    duration = float(duration_s)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_k must be finite and positive")
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration_s must be finite and non-negative")
    if method not in {"BDF", "Radau"}:
        raise ValueError("kinetics method must be BDF or Radau")
    if relative_tolerance <= 0 or absolute_tolerance_mol_m3 <= 0:
        raise ValueError("kinetics tolerances must be positive")

    network, species, stoich, sparsity = _prepare_network(
        reactions, initial_concentration_mol_m3, database
    )
    index = {key: i for i, key in enumerate(species)}
    y0 = np.zeros(len(species), dtype=float)
    for raw_key, value in initial_concentration_mol_m3.items():
        y0[index[canonical_species(raw_key)]] += float(value)

    if sample_times_s is None:
        t_eval = np.asarray([0.0, duration], dtype=float) if duration > 0 else np.asarray([0.0])
    else:
        t_eval = np.asarray(sample_times_s, dtype=float)
        if t_eval.ndim != 1 or t_eval.size == 0 or np.any(~np.isfinite(t_eval)):
            raise ValueError("sample_times_s must be a non-empty finite 1-D array")
        if np.any(np.diff(t_eval) < 0) or t_eval[0] < 0 or t_eval[-1] > duration:
            raise ValueError("sample_times_s must be sorted and lie within [0, duration_s]")
        if t_eval[0] != 0.0:
            t_eval = np.concatenate(([0.0], t_eval))
        if duration > 0 and t_eval[-1] != duration:
            t_eval = np.concatenate((t_eval, [duration]))

    photolysis = dict(photolysis_rates_s1 or {})

    def rhs(_time: float, raw_state: np.ndarray) -> np.ndarray:
        state = np.maximum(np.asarray(raw_state, dtype=float), 0.0)
        concentrations = {key: float(state[i]) for i, key in enumerate(species)}
        rates_mapping = reaction_rates(
            reactions=network,
            concentration_mol_m3=concentrations,
            temperature_k=temperature,
            photolysis_rates_s1=photolysis,
        )
        rates_vector = np.asarray([rates_mapping[item.key] for item in network], dtype=float)
        return stoich @ rates_vector

    if duration == 0:
        time = np.asarray([0.0])
        values = y0[:, None]
        nfev = njev = nlu = 0
        success = True
        message = "zero-duration integration"
    else:
        solution = solve_ivp(
            rhs,
            (0.0, duration),
            y0,
            method=method,
            t_eval=t_eval,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance_mol_m3),
            jac_sparsity=sparsity,
        )
        time = np.asarray(solution.t, dtype=float)
        values = np.asarray(solution.y, dtype=float)
        nfev = int(solution.nfev)
        njev = int(getattr(solution, "njev", 0) or 0)
        nlu = int(getattr(solution, "nlu", 0) or 0)
        success = bool(solution.success)
        message = str(solution.message)

    minimum_raw = float(np.min(values)) if values.size else 0.0
    negative_limit = max(50.0 * float(absolute_tolerance_mol_m3), 1.0e-12)
    if minimum_raw < -negative_limit:
        raise RuntimeError(
            f"kinetics solver produced significant negative concentration {minimum_raw:g} mol m^-3"
        )
    values = np.maximum(values, 0.0)
    final_vector = values[:, -1]
    final = {key: float(final_vector[i]) for i, key in enumerate(species)}
    before = _element_totals({key: float(y0[i]) for i, key in enumerate(species)}, database)
    after = _element_totals(final, database)
    element_residual = _relative_element_residual(before, after)
    if not success:
        raise RuntimeError(f"kinetics solver failed: {message}")
    if element_residual > max(2.0e-7, 20.0 * relative_tolerance):
        raise RuntimeError(f"kinetics elemental closure failed: {element_residual:g}")

    final_rates = reaction_rates(
        reactions=network,
        concentration_mol_m3=final,
        temperature_k=temperature,
        photolysis_rates_s1=photolysis,
    )
    trajectories = {key: values[i].copy() for i, key in enumerate(species)}
    return KineticsResult(
        time_s=time.copy(),
        concentration_mol_m3=trajectories,
        final_concentration_mol_m3=final,
        reaction_rates_mol_m3_s=final_rates,
        converged=True,
        method=method,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
        minimum_raw_concentration_mol_m3=minimum_raw,
        element_relative_residual=element_residual,
        message=message,
    )
