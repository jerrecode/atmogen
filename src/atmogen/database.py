from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping

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
    longwave_mass_absorption_m2_kg_fast: float
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


class ChemicalDatabase:
    """Versioned, immutable scientific-data view used by solver backends."""

    def __init__(self, species: Mapping[str, Species]) -> None:
        self._species = dict(species)
        if len(self._species) != len(set(self._species)):
            raise ValueError("duplicate species keys")

    @property
    def species(self) -> Mapping[str, Species]:
        return self._species

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
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(raw).hexdigest()


_ALIASES = {"water": "H2O", "carbon_dioxide": "CO2", "methane": "CH4", "nitrogen": "N2", "oxygen": "O2", "argon": "Ar", "ammonia": "NH3", "hydrogen": "H2", "helium": "He", "carbon_monoxide": "CO"}


def canonical_species(key: str) -> str:
    stripped = str(key).strip()
    return _ALIASES.get(stripped.lower(), stripped)


def _sp(key: str, formula: Mapping[str, int], mm: float, phase: str, hf: float, s: float, cp: float,
        freeze: float | None, critical: float | None, rho: float | None, latent: float | None,
        rayleigh: float | None, lw: float, provenance: ProvenanceClass = ProvenanceClass.FITTED,
        validity: str = "FAST constant-Cp continuation; validate before high-fidelity use") -> Species:
    return Species(key, formula, mm, phase, hf, s, cp, freeze, critical, rho, latent, rayleigh, lw,
                   provenance, "NIST Chemistry WebBook thermochemistry; optical FAST coefficients separately estimated", validity)


# Thermochemistry is compact validation data, not a replacement for temperature-
# segmented NASA/Shomate tables. Optical longwave coefficients are explicitly
# ESTIMATED screening coefficients and never exposed as measured constants.
BUILTIN_DATABASE = ChemicalDatabase({
    "H2": _sp("H2", {"H": 2}, 2.01588e-3, "gas", 0.0, 130.68, 28.84, 13.99, 33.15, 70.8, 4.46e5, 8.5e-32, 1e-5),
    "He": _sp("He", {"He": 1}, 4.002602e-3, "gas", 0.0, 126.15, 20.79, None, 5.20, 125.0, 2.1e4, 1.0e-32, 0.0),
    "N2": _sp("N2", {"N": 2}, 28.0134e-3, "gas", 0.0, 191.61, 29.12, 63.15, 126.19, 808.0, 1.99e5, 5.1e-31, 2e-5),
    "O2": _sp("O2", {"O": 2}, 31.9988e-3, "gas", 0.0, 205.15, 29.38, 54.36, 154.58, 1141.0, 2.13e5, 4.8e-31, 2e-5),
    "Ar": _sp("Ar", {"Ar": 1}, 39.948e-3, "gas", 0.0, 154.85, 20.79, 83.81, 150.69, 1395.0, 1.61e5, 4.6e-31, 0.0),
    "CO": _sp("CO", {"C": 1, "O": 1}, 28.0101e-3, "gas", -110530.0, 197.66, 29.14, 68.15, 132.86, 789.0, 2.16e5, 8.0e-31, 0.006),
    "CO2": _sp("CO2", {"C": 1, "O": 2}, 44.0095e-3, "gas", -393510.0, 213.79, 37.14, 216.58, 304.13, 1100.0, 5.74e5, 1.24e-30, 0.08),
    "CH4": _sp("CH4", {"C": 1, "H": 4}, 16.0425e-3, "gas", -74870.0, 186.25, 35.69, 90.69, 190.56, 422.0, 5.10e5, 1.6e-30, 0.04),
    "H2O": _sp("H2O", {"H": 2, "O": 1}, 18.01528e-3, "gas", -241826.0, 188.84, 33.58, 273.16, 647.096, 997.0, 2.50e6, 2.5e-31, 0.12,
                  validity="Thermochemistry: FAST constant-Cp continuation; saturation: IAPWS form in valid range"),
    "NH3": _sp("NH3", {"N": 1, "H": 3}, 17.03052e-3, "gas", -45940.0, 192.77, 35.06, 195.40, 405.40, 682.0, 1.37e6, 1.2e-30, 0.06),
    "C_graphite": _sp("C_graphite", {"C": 1}, 12.0107e-3, "solid", 0.0, 5.74, 8.53, None, None, 2260.0, None, None, 0.0),
})


def species_moles_to_elements(species_moles: Mapping[str, float], database: ChemicalDatabase = BUILTIN_DATABASE) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, amount in species_moles.items():
        sp = database.get(key)
        for element, count in sp.formula.items():
            result[element] = result.get(element, 0.0) + float(amount) * count
    return result
