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


@dataclass(frozen=True, slots=True)
class LiquidMixtureTransportFields:
    """Vectorized screening properties with an explicit validity mask.

    Property arrays are zero where active_mask is false. Consumers must use the
    mask rather than interpreting those zeros as physical liquid properties.
    """

    density_kg_m3: np.ndarray
    dynamic_viscosity_pa_s: np.ndarray
    surface_tension_n_m: np.ndarray
    total_mass_kg: np.ndarray
    active_mask: np.ndarray
    mass_fractions: Mapping[str, np.ndarray]
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


def liquid_mixture_transport_fields(
    *,
    species_mass_kg: Mapping[str, np.ndarray],
    database: ChemicalDatabase = BUILTIN_DATABASE,
    transport_database: Mapping[str, FluidTransportProperties] = BUILTIN_FLUID_TRANSPORT,
    include_mass_fractions: bool = True,
) -> LiquidMixtureTransportFields | None:
    """Vectorize the scalar screening mixture law over equal-shaped mass fields.

    Species aliases are canonicalized and duplicate aliases are added together.
    Every mass field must be finite, non-negative, and have the same shape. If no
    cell contains positive liquid mass, or if any species with positive mass lacks
    transport data, None is returned. Property arrays are exactly zero in inactive
    cells and active_mask identifies cells where the values are meaningful.

    include_mass_fractions=False is a host-facing compact mode for large spatial
    grids. It returns the same mixture properties, total mass, active mask, component
    provenance and method semantics while leaving mass_fractions empty instead of
    retaining one full-size array per species.
    """
    if not isinstance(include_mass_fractions, bool):
        raise TypeError("include_mass_fractions must be a boolean")
    if not species_mass_kg:
        return None

    canonical_mass: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for raw_key, raw_values in species_mass_kg.items():
        key = canonical_species(raw_key)
        values = np.asarray(raw_values, dtype=np.float64)
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError(
                f"species mass fields must have one shape; {key!r} has "
                f"{values.shape}, expected {shape}"
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(
                f"species mass field {raw_key!r} must be finite and non-negative"
            )
        if key in canonical_mass:
            canonical_mass[key] = canonical_mass[key] + values
        else:
            # The solver never mutates component input fields, so a unique
            # canonical float64 array can be referenced directly. Duplicate aliases
            # still allocate their explicit sum above.
            canonical_mass[key] = values

    assert shape is not None
    positive_species = {
        key for key, values in canonical_mass.items() if np.any(values > 0.0)
    }
    if not positive_species:
        return None

    component: dict[str, FluidTransportProperties] = {}
    for key in sorted(positive_species):
        props = transport_database.get(key)
        if props is None:
            return None
        database.get(key)
        component[key] = props

    total_mass = np.zeros(shape, dtype=np.float64)
    for key in positive_species:
        total_mass += canonical_mass[key]
    active = total_mass > 0.0

    fractions: dict[str, np.ndarray] = {}
    specific_volume = np.zeros(shape, dtype=np.float64)
    log_viscosity = np.zeros(shape, dtype=np.float64)
    tension = np.zeros(shape, dtype=np.float64)
    compact_fraction = (
        np.zeros(shape, dtype=np.float64)
        if not include_mass_fractions
        else None
    )
    for key in sorted(positive_species):
        if include_mass_fractions:
            fraction = np.divide(
                canonical_mass[key],
                total_mass,
                out=np.zeros(shape, dtype=np.float64),
                where=active,
            )
            fractions[key] = fraction
        else:
            assert compact_fraction is not None
            compact_fraction.fill(0.0)
            fraction = np.divide(
                canonical_mass[key],
                total_mass,
                out=compact_fraction,
                where=active,
            )
        props = component[key]
        specific_volume += fraction / props.density_kg_m3
        log_viscosity += fraction * np.log(props.dynamic_viscosity_pa_s)
        tension += fraction * props.surface_tension_n_m

    density = np.divide(
        1.0,
        specific_volume,
        out=np.zeros(shape, dtype=np.float64),
        where=active,
    )
    viscosity = np.zeros(shape, dtype=np.float64)
    viscosity[active] = np.exp(log_viscosity[active])
    tension[~active] = 0.0

    return LiquidMixtureTransportFields(
        density_kg_m3=density,
        dynamic_viscosity_pa_s=viscosity,
        surface_tension_n_m=tension,
        total_mass_kg=total_mass,
        active_mask=active,
        mass_fractions=fractions,
        component_properties=component,
        method=(
            "vectorized ideal-volume density + mass-fraction log-viscosity + "
            "linear surface-tension screening blend"
            + (
                "; mass-fraction fields omitted by compact mode"
                if not include_mass_fractions
                else ""
            )
        ),
    )


__all__ = [
    "BUILTIN_FLUID_TRANSPORT",
    "FluidTransportProperties",
    "LiquidMixtureTransportFields",
    "LiquidMixtureTransportProperties",
    "fluid_transport_properties",
    "liquid_mixture_transport_fields",
    "liquid_mixture_transport_properties",
]
