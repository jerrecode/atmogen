from __future__ import annotations

"""Screening-grade liquid transport properties for planetary surface coupling.

This module deliberately stays separate from the thermochemical Species schema.
Values are reference properties for reduced-order host-model scaling rather than
temperature/pressure-dependent high-fidelity material fits.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .database import BUILTIN_DATABASE, ChemicalDatabase, ProvenanceClass, canonical_species


@dataclass(frozen=True, slots=True)
class FluidTransportProperties:
    species: str
    reference_temperature_k: float
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    surface_tension_n_m: float
    freezing_temperature_k: float | None
    provenance_class: ProvenanceClass
    source: str
    validity: str

    def __post_init__(self) -> None:
        for name in (
            "reference_temperature_k",
            "density_kg_m3",
            "dynamic_viscosity_pa_s",
            "surface_tension_n_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.freezing_temperature_k is not None and (
            not np.isfinite(self.freezing_temperature_k)
            or self.freezing_temperature_k <= 0
        ):
            raise ValueError("freezing_temperature_k must be positive when supplied")
        if not self.source or not self.validity:
            raise ValueError("transport-property provenance and validity are required")


@dataclass(frozen=True, slots=True)
class LiquidMixtureTransportProperties:
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    surface_tension_n_m: float
    mass_fractions: Mapping[str, float]
    component_properties: Mapping[str, FluidTransportProperties]
    method: str


def _screening(
    species: str,
    reference_temperature_k: float,
    viscosity_mpa_s: float,
    surface_tension_mn_m: float,
    *,
    density_kg_m3: float | None = None,
    validity: str,
) -> FluidTransportProperties:
    key = canonical_species(species)
    thermo = BUILTIN_DATABASE.get(key)
    density = thermo.liquid_density_kg_m3 if density_kg_m3 is None else density_kg_m3
    if density is None or density <= 0:
        raise ValueError(f"{key} has no positive liquid density in the bundled database")
    return FluidTransportProperties(
        species=key,
        reference_temperature_k=float(reference_temperature_k),
        density_kg_m3=float(density),
        dynamic_viscosity_pa_s=float(viscosity_mpa_s) * 1.0e-3,
        surface_tension_n_m=float(surface_tension_mn_m) * 1.0e-3,
        freezing_temperature_k=thermo.freezing_point_k,
        provenance_class=ProvenanceClass.ESTIMATED,
        source=(
            "atmogen screening reference set compiled for reduced-order planetary "
            "surface-process scaling; density/freezing data originate from the "
            "versioned built-in chemical database"
        ),
        validity=validity,
    )


BUILTIN_FLUID_TRANSPORT: Mapping[str, FluidTransportProperties] = {
    "H2O": _screening(
        "H2O", 293.15, 1.002, 72.0,
        validity="screening reference near ambient liquid water; no T/P dependence",
    ),
    "CH4": _screening(
        "CH4", 91.0, 0.117, 17.0,
        validity="screening reference near methane normal-boiling regime; no T/P dependence",
    ),
    "C2H6": _screening(
        "C2H6", 94.0, 0.24, 16.0,
        validity="screening reference for cold liquid ethane; no T/P dependence",
    ),
    "NH3": _screening(
        "NH3", 240.0, 0.25, 20.0,
        validity="screening reference for liquid ammonia; no T/P dependence",
    ),
    "N2": _screening(
        "N2", 77.0, 0.16, 8.9,
        validity="screening reference for cryogenic liquid nitrogen; no T/P dependence",
    ),
    "O2": _screening(
        "O2", 90.0, 0.20, 13.2,
        validity="screening reference for cryogenic liquid oxygen; no T/P dependence",
    ),
    "Ar": _screening(
        "Ar", 87.0, 0.27, 13.0,
        validity="screening reference for cryogenic liquid argon; no T/P dependence",
    ),
    "CO": _screening(
        "CO", 82.0, 0.17, 9.0,
        validity="screening reference for cryogenic liquid carbon monoxide; no T/P dependence",
    ),
    "CO2": _screening(
        "CO2", 220.0, 0.10, 18.0,
        validity="screening extrapolation for dense/liquid CO2 regimes; no T/P dependence",
    ),
    "SO2": _screening(
        "SO2", 270.0, 0.40, 33.0,
        validity="screening reference for liquid sulfur dioxide; no T/P dependence",
    ),
}


def fluid_transport_properties(species: str) -> FluidTransportProperties | None:
    """Return a supported screening-grade liquid transport-property record."""
    return BUILTIN_FLUID_TRANSPORT.get(canonical_species(species))


def liquid_mixture_transport_properties(
    *,
    species_mass_kg: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
    transport_database: Mapping[str, FluidTransportProperties] = BUILTIN_FLUID_TRANSPORT,
) -> LiquidMixtureTransportProperties | None:
    """Return ideal-volume/log-viscosity screening properties for a liquid mixture.

    Density uses ideal additive component volumes. Dynamic viscosity uses a
    mass-fraction logarithmic blend. Surface tension uses a mass-fraction linear
    blend. Missing component transport data yields None rather than an invented
    fallback.
    """
    positive: dict[str, float] = {}
    for raw_key, raw_mass in species_mass_kg.items():
        key = canonical_species(raw_key)
        mass = float(raw_mass)
        if not np.isfinite(mass) or mass < 0:
            raise ValueError(f"invalid species mass {raw_key!r}: {raw_mass!r}")
        if mass > 0:
            positive[key] = positive.get(key, 0.0) + mass
    if not positive:
        return None

    component: dict[str, FluidTransportProperties] = {}
    for key in positive:
        props = transport_database.get(key)
        if props is None:
            return None
        database.get(key)
        component[key] = props

    total_mass = float(sum(positive.values()))
    fractions = {key: mass / total_mass for key, mass in positive.items()}
    specific_volume = sum(
        fractions[key] / component[key].density_kg_m3 for key in fractions
    )
    density = 1.0 / max(specific_volume, 1.0e-300)
    log_viscosity = sum(
        fractions[key] * np.log(component[key].dynamic_viscosity_pa_s)
        for key in fractions
    )
    viscosity = float(np.exp(log_viscosity))
    tension = float(sum(
        fractions[key] * component[key].surface_tension_n_m for key in fractions
    ))
    return LiquidMixtureTransportProperties(
        density_kg_m3=float(density),
        dynamic_viscosity_pa_s=viscosity,
        surface_tension_n_m=tension,
        mass_fractions=fractions,
        component_properties=component,
        method="ideal-volume density + mass-fraction log-viscosity + linear surface-tension screening blend",
    )


__all__ = [
    "BUILTIN_FLUID_TRANSPORT",
    "FluidTransportProperties",
    "LiquidMixtureTransportProperties",
    "fluid_transport_properties",
    "liquid_mixture_transport_properties",
]
