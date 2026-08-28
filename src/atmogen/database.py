from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np

from .version import DATA_SCHEMA_VERSION


class ProvenanceClass(str, Enum):
    MEASURED = "MEASURED"
    FITTED = "FITTED"
    THEORETICAL = "THEORETICAL"
    GROUP_CONTRIBUTION = "GROUP_CONTRIBUTION"
    INTERPOLATED = "INTERPOLATED"
    EXTRAPOLATED = "EXTRAPOLATED"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Species:
    key: str
    formula: Mapping[str, int]
    molar_mass_kg_mol: float
    phase: str
    formation_enthalpy_j_mol_298: float
    entropy_j_mol_k_298: float
    heat_capacity_j_mol_k: float
    freezing_point_k: float | None
    critical_temperature_k: float | None
    liquid_density_kg_m3: float | None
    latent_heat_j_kg: float | None
    rayleigh_cross_section_550_m2: float | None
    longwave_column_coefficient_fast: float
    provenance_class: ProvenanceClass
    source: str
    validity: str

    def standard_gibbs_j_mol(self, temperature_k: float) -> float:
        """Constant-Cp continuation from 298.15 K; intentionally a FAST approximation."""
        t = float(temperature_k)
        t0 = 298.15
        h = self.formation_enthalpy_j_mol_298 + self.heat_capacity_j_mol_k * (t - t0)
        s = self.entropy_j_mol_k_298 + self.heat_capacity_j_mol_k * np.log(t / t0)
        return float(h - t * s)


@dataclass(frozen=True, slots=True)
class NRTLInteraction:
    """Directed binary interaction for an isothermal NRTL activity model."""

    component_i: str
    component_j: str
    delta_g_ij_j_mol: float
    alpha: float
    provenance_class: ProvenanceClass
    source: str
    validity: str

    def __post_init__(self) -> None:
        if not self.component_i or not self.component_j or self.component_i == self.component_j:
            raise ValueError("NRTL interaction requires two distinct component keys")
        if not np.isfinite(self.delta_g_ij_j_mol):
            raise ValueError("delta_g_ij_j_mol must be finite")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("NRTL alpha must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Reaction:
    """Stoichiometric kinetic reaction with explicit rate-law/provenance semantics.

    Concentrations are interpreted as mol m^-3 and volumetric reaction rates as
    mol m^-3 s^-1. For ``rate_law='arrhenius'`` the pre-exponential factor therefore
    has units ``(m^3 mol^-1)^(order-1) s^-1`` where ``order`` is the sum of reactant
    stoichiometric exponents. ``rate_coefficient_units`` records the exact external
    unit text used by the scientific source and is never inferred as provenance.

    ``rate_law='photolysis'`` represents one-parent first-order photodestruction.
    Its local J value [s^-1] is supplied by the photochemistry solver rather than
    embedded in this record.
    """

    key: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_law: str = "arrhenius"  # arrhenius | photolysis
    pre_exponential_factor_si: float | None = None
    temperature_exponent: float = 0.0
    activation_energy_j_mol: float = 0.0
    reference_temperature_k: float = 298.15
    rate_coefficient_units: str = ""
    provenance_class: ProvenanceClass = ProvenanceClass.UNKNOWN
    source: str = ""
    validity: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("reaction key cannot be empty")
        if self.rate_law not in {"arrhenius", "photolysis"}:
            raise ValueError("reaction rate_law must be arrhenius or photolysis")
        if not self.reactants or not self.products:
            raise ValueError("reaction requires reactants and products")
        for label, mapping in (("reactant", self.reactants), ("product", self.products)):
            for species, coefficient in mapping.items():
                value = float(coefficient)
                if not species or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {label} stoichiometry {species!r}: {coefficient!r}")
        if not np.isfinite(self.temperature_exponent):
            raise ValueError("temperature_exponent must be finite")
        if not np.isfinite(self.activation_energy_j_mol):
            raise ValueError("activation_energy_j_mol must be finite")
        if not np.isfinite(self.reference_temperature_k) or self.reference_temperature_k <= 0:
            raise ValueError("reference_temperature_k must be finite and positive")
        if self.rate_law == "arrhenius":
            if self.pre_exponential_factor_si is None or not np.isfinite(self.pre_exponential_factor_si) or self.pre_exponential_factor_si < 0:
                raise ValueError("arrhenius reaction requires a finite non-negative pre_exponential_factor_si")
        else:
            if len(self.reactants) != 1 or abs(float(next(iter(self.reactants.values()))) - 1.0) > 1e-12:
                raise ValueError("photolysis reaction must contain exactly one parent reactant with coefficient 1")
            if self.pre_exponential_factor_si not in {None, 0.0}:
                raise ValueError("photolysis J is supplied externally; pre_exponential_factor_si must be None or zero")

    @property
    def order(self) -> float:
        return float(sum(float(value) for value in self.reactants.values()))


class ChemicalDatabase:
    """Versioned, immutable scientific-data view used by solver backends."""

    def __init__(self, species: Mapping[str, Species],
                 nrtl_interactions: Iterable[NRTLInteraction] = (),
                 reactions: Iterable[Reaction] = ()) -> None:
        self._species = dict(species)
        if len(self._species) != len(set(self._species)):
            raise ValueError("duplicate species keys")
        interactions: dict[tuple[str, str], NRTLInteraction] = {}
        for item in nrtl_interactions:
            key = (canonical_species(item.component_i), canonical_species(item.component_j))
            if key in interactions:
                raise ValueError(f"duplicate NRTL interaction {key!r}")
            if key[0] not in self._species or key[1] not in self._species:
                raise ValueError(f"NRTL interaction references unknown species {key!r}")
            interactions[key] = NRTLInteraction(
                key[0], key[1], item.delta_g_ij_j_mol, item.alpha,
                item.provenance_class, item.source, item.validity,
            )
        self._nrtl_interactions = interactions

        reaction_map: dict[str, Reaction] = {}
        for item in reactions:
            if item.key in reaction_map:
                raise ValueError(f"duplicate reaction key {item.key!r}")
            reactants = {canonical_species(key): float(value) for key, value in item.reactants.items()}
            products = {canonical_species(key): float(value) for key, value in item.products.items()}
            for key in set(reactants) | set(products):
                if key not in self._species:
                    raise ValueError(f"reaction {item.key!r} references unknown species {key!r}")
            canonical = Reaction(
                key=item.key,
                reactants=reactants,
                products=products,
                rate_law=item.rate_law,
                pre_exponential_factor_si=item.pre_exponential_factor_si,
                temperature_exponent=item.temperature_exponent,
                activation_energy_j_mol=item.activation_energy_j_mol,
                reference_temperature_k=item.reference_temperature_k,
                rate_coefficient_units=item.rate_coefficient_units,
                provenance_class=item.provenance_class,
                source=item.source,
                validity=item.validity,
            )
            residual = reaction_element_residual(canonical, self._species)
            if residual and max(abs(value) for value in residual.values()) > 1e-10:
                raise ValueError(f"reaction {item.key!r} is not element-balanced: {residual}")
            reaction_map[item.key] = canonical
        self._reactions = reaction_map

    @property
    def species(self) -> Mapping[str, Species]:
        return self._species

    @property
    def nrtl_interactions(self) -> Mapping[tuple[str, str], NRTLInteraction]:
        return self._nrtl_interactions

    @property
    def reactions(self) -> Mapping[str, Reaction]:
        return self._reactions

    def get(self, key: str) -> Species:
        canonical = canonical_species(key)
        try:
            return self._species[canonical]
        except KeyError as exc:
            raise KeyError(f"species {key!r} is not in database") from exc

    @property
    def revision_hash(self) -> str:
        payload = {
            "schema": DATA_SCHEMA_VERSION,
            "species": {key: asdict(value) for key, value in sorted(self._species.items())},
            "nrtl_interactions": {
                f"{key[0]}->{key[1]}": asdict(value)
                for key, value in sorted(self._nrtl_interactions.items())
            },
            "reactions": {key: asdict(value) for key, value in sorted(self._reactions.items())},
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(raw).hexdigest()


_ALIASES = {"water": "H2O", "carbon_dioxide": "CO2", "methane": "CH4", "nitrogen": "N2", "oxygen": "O2", "argon": "Ar", "ammonia": "NH3", "hydrogen": "H2", "helium": "He", "carbon_monoxide": "CO"}


def canonical_species(key: str) -> str:
    stripped = str(key).strip()
    return _ALIASES.get(stripped.lower(), stripped)


def reaction_element_residual(reaction: Reaction, species: Mapping[str, Species]) -> dict[str, float]:
    """Products minus reactants in stoichiometric element counts."""
    residual: dict[str, float] = {}
    for sign, side in ((-1.0, reaction.reactants), (1.0, reaction.products)):
        for raw_key, coefficient in side.items():
            key = canonical_species(raw_key)
            if key not in species:
                raise KeyError(f"reaction {reaction.key!r} references unknown species {key!r}")
            for element, count in species[key].formula.items():
                residual[element] = residual.get(element, 0.0) + sign * float(coefficient) * float(count)
    return {key: float(value) for key, value in residual.items() if abs(value) > 1e-14}


def _sp(key: str, formula: Mapping[str, int], mm: float, phase: str, hf: float, s: float, cp: float,
        freeze: float | None, critical: float | None, rho: float | None, latent: float | None,
        rayleigh: float | None, lw: float, provenance: ProvenanceClass = ProvenanceClass.FITTED,
        validity: str = "FAST constant-Cp continuation; validate before high-fidelity use") -> Species:
    return Species(key, formula, mm, phase, hf, s, cp, freeze, critical, rho, latent, rayleigh, lw,
                   provenance, "NIST Chemistry WebBook thermochemistry; optical FAST coefficients separately estimated", validity)


# Thermochemistry is compact validation data, not a replacement for temperature-
# segmented NASA/Shomate tables. Optical longwave coefficients are explicitly
# ESTIMATED square-root-column screening coefficients and never exposed as
# measured mass-absorption constants. NRTL parameters and kinetic reactions are
# intentionally absent from the bundled validation database until sources and
# validity ranges have been explicitly checked.
BUILTIN_DATABASE = ChemicalDatabase({
    "H2": _sp("H2", {"H": 2}, 2.01588e-3, "gas", 0.0, 130.68, 28.84, 13.99, 33.15, 70.8, 4.46e5, 8.5e-32, 1e-5),
    "He": _sp("He", {"He": 1}, 4.002602e-3, "gas", 0.0, 126.15, 20.79, None, 5.20, 125.0, 2.1e4, 1.0e-32, 0.0),
    "N2": _sp("N2", {"N": 2}, 28.0134e-3, "gas", 0.0, 191.61, 29.12, 63.15, 126.19, 808.0, 1.99e5, 5.1e-31, 2e-5),
    "O2": _sp("O2", {"O": 2}, 31.9988e-3, "gas", 0.0, 205.15, 29.38, 54.36, 154.58, 1141.0, 2.13e5, 4.8e-31, 2e-5),
    "Ar": _sp("Ar", {"Ar": 1}, 39.948e-3, "gas", 0.0, 154.85, 20.79, 83.81, 150.69, 1395.0, 1.61e5, 4.6e-31, 0.0),
    "CO": _sp("CO", {"C": 1, "O": 1}, 28.0101e-3, "gas", -110530.0, 197.66, 29.14, 68.15, 132.86, 789.0, 2.16e5, 8.0e-31, 0.006),
    "CO2": _sp("CO2", {"C": 1, "O": 2}, 44.0095e-3, "gas", -393510.0, 213.79, 37.14, 216.58, 304.13, 1100.0, 5.74e5, 1.24e-30, 0.08),
    "CH4": _sp("CH4", {"C": 1, "H": 4}, 16.0425e-3, "gas", -74870.0, 186.25, 35.69, 90.69, 190.56, 422.0, 5.10e5, 1.6e-30, 0.04),
    "C2H6": _sp("C2H6", {"C": 2, "H": 6}, 30.0690e-3, "gas", -84680.0, 229.49, 52.49, 90.35, 305.32, 544.0, 4.89e5, 2.8e-30, 0.035),
    "H2O": _sp("H2O", {"H": 2, "O": 1}, 18.01528e-3, "gas", -241826.0, 188.84, 33.58, 273.16, 647.096, 997.0, 2.50e6, 2.5e-31, 0.12,
                  validity="Thermochemistry: FAST constant-Cp continuation; saturation: IAPWS form in valid range"),
    "NH3": _sp("NH3", {"N": 1, "H": 3}, 17.03052e-3, "gas", -45940.0, 192.77, 35.06, 195.40, 405.40, 682.0, 1.37e6, 1.2e-30, 0.06),
    "SO2": _sp("SO2", {"S": 1, "O": 2}, 64.066e-3, "gas", -296840.0, 248.22, 39.87, 197.67, 430.64, 1434.0, 3.89e5, 3.0e-30, 0.10),
    "C_graphite": _sp("C_graphite", {"C": 1}, 12.0107e-3, "solid", 0.0, 5.74, 8.53, None, None, 2260.0, None, None, 0.0),
})


def species_moles_to_elements(species_moles: Mapping[str, float], database: ChemicalDatabase = BUILTIN_DATABASE) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, amount in species_moles.items():
        sp = database.get(key)
        for element, count in sp.formula.items():
            result[element] = result.get(element, 0.0) + float(amount) * count
    return result
