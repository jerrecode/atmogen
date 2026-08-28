from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .database import BUILTIN_DATABASE, ChemicalDatabase

R_GAS = 8.31446261815324
STANDARD_PRESSURE_PA = 1e5


@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    species_moles: Mapping[str, float]
    gas_mole_fractions: Mapping[str, float]
    element_relative_residual: float
    gibbs_j: float
    converged: bool
    iterations: int
    active_species: tuple[str, ...]
    pruned_species: tuple[str, ...]
    backend: str
    message: str


class EquilibriumBackend(Protocol):
    def solve(self, *, temperature_k: float, pressure_pa: float, element_moles: Mapping[str, float],
              initial_species_moles: Mapping[str, float] | None = None) -> EquilibriumResult: ...


class IdealGibbsEquilibrium:
    """Element-constrained ideal-mixture Gibbs minimizer for compact networks.

    This backend is intended for FAST/STANDARD validation networks. It independently
    checks element closure and does not interpret an optimizer success flag as proof
    of physical validity.
    """

    def __init__(self, database: ChemicalDatabase = BUILTIN_DATABASE) -> None:
        self.database = database

    def solve(self, *, temperature_k: float, pressure_pa: float, element_moles: Mapping[str, float],
              initial_species_moles: Mapping[str, float] | None = None) -> EquilibriumResult:
        if temperature_k <= 0 or pressure_pa <= 0:
            raise ValueError("temperature and pressure must be positive")
        elements = tuple(sorted(k for k, v in element_moles.items() if v > 0))
        if not elements:
            raise ValueError("element inventory has no positive amounts")
        candidates = [sp for sp in self.database.species.values()
                      if set(sp.formula).issubset(elements) and sp.formula]
        # Species containing an available element but requiring a missing one cannot
        # be admitted; species with no route to any element are thereby pruned.
        pruned = tuple(sorted(set(self.database.species) - {sp.key for sp in candidates}))
        if not candidates:
            raise ValueError(f"no candidate species can represent elements {elements}")
        a = np.asarray([[sp.formula.get(el, 0) for sp in candidates] for el in elements], float)
        b = np.asarray([element_moles[el] for el in elements], float)
        scale = max(float(np.max(b)), 1e-30)
        bs = b / scale
        floor = 1e-16

        x0 = np.full(len(candidates), floor)
        hints = initial_species_moles or {}
        for idx, sp in enumerate(candidates):
            x0[idx] = max(float(hints.get(sp.key, 0.0)) / scale, floor)
        # Find a non-negative element-feasible starting point if hints are absent or inconsistent.
        feasibility = minimize(lambda x: float(np.sum(x)), x0, method="SLSQP",
                               bounds=[(0.0, None)] * len(candidates),
                               constraints=[LinearConstraint(a, bs, bs)],
                               options={"ftol": 1e-12, "maxiter": 600})
        if feasibility.success:
            x0 = np.maximum(feasibility.x, floor)

        g0 = np.asarray([sp.standard_gibbs_j_mol(temperature_k) for sp in candidates], float)
        gas = np.asarray([sp.phase == "gas" for sp in candidates])

        def objective(x: np.ndarray) -> float:
            safe = np.maximum(x, floor)
            gas_total = max(float(np.sum(safe[gas])), floor)
            mu = g0.copy()
            if np.any(gas):
                activity = safe[gas] / gas_total * pressure_pa / STANDARD_PRESSURE_PA
                mu[gas] += R_GAS * temperature_k * np.log(np.maximum(activity, floor))
            return float(np.dot(safe, mu) / (R_GAS * temperature_k))

        result = minimize(objective, x0, method="SLSQP", bounds=[(0.0, None)] * len(candidates),
                          constraints=[LinearConstraint(a, bs, bs)],
                          options={"ftol": 1e-11, "maxiter": 1000})
        x = np.maximum(np.asarray(result.x, float), 0.0) * scale
        residual = a @ (x / scale) - bs
        rel = float(np.max(np.abs(residual) / np.maximum(np.abs(bs), 1e-14)))
        amounts = {sp.key: float(value) for sp, value in zip(candidates, x, strict=True)
                   if value > scale * 1e-13}
        gas_amounts = {key: value for key, value in amounts.items() if self.database.get(key).phase == "gas"}
        total_gas = sum(gas_amounts.values())
        fractions = {key: value / total_gas for key, value in gas_amounts.items()} if total_gas > 0 else {}
        converged = bool(result.success and rel <= 2e-8 and all(v >= 0 for v in amounts.values()))
        return EquilibriumResult(amounts, fractions, rel, objective(np.asarray(result.x)) * R_GAS * temperature_k * scale,
                                 converged, int(getattr(result, "nit", 0)), tuple(sp.key for sp in candidates),
                                 pruned, "scipy-slsqp-ideal-gibbs-v1", str(result.message))


def normalized_initial_composition(initial_species_moles: Mapping[str, float]) -> dict[str, float]:
    positive = {str(k): float(v) for k, v in initial_species_moles.items() if float(v) > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("fixed_species chemistry requires a positive initial molecular state")
    return {key: value / total for key, value in positive.items()}
