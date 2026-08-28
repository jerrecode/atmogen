from __future__ import annotations

from typing import Mapping

import numpy as np

from .database import BUILTIN_DATABASE, ChemicalDatabase
from .models import PhaseReservoirResult, PlanetPhysicalState, SurfaceReservoirs


def saturation_pressure_pa(species: str, temperature_k: float) -> tuple[float | None, str | None]:
    """Return vapor pressure and provenance/fallback note.

    Water uses the IAPWS saturation release form between the triple and critical
    points. Other bundled condensables use bounded Clausius-Clapeyron screening
    fits and are labelled as estimates.
    """
    t = float(temperature_k)
    if species == "H2O" and 273.16 <= t <= 647.096:
        tc, pc = 647.096, 22.064e6
        theta = 1.0 - t / tc
        a = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
        powers = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
        return float(pc * np.exp(tc / t * sum(c * theta**p for c, p in zip(a, powers, strict=True)))), None
    anchors = {
        "CO2": (216.58, 5.18e5, 5.74e5, 44.0095e-3),
        "CH4": (90.69, 1.17e4, 5.10e5, 16.0425e-3),
        "NH3": (195.40, 6.06e3, 1.37e6, 17.03052e-3),
        "C2H6": (90.35, 1.1, 4.89e5, 30.0690e-3),
        "SO2": (197.67, 1.67e4, 3.89e5, 64.066e-3),
    }
    if species in anchors:
        tref, pref, latent, mm = anchors[species]
        exponent = np.clip(-latent * mm / 8.31446261815324 * (1.0 / t - 1.0 / tref), -745.0, 700.0)
        value = pref * np.exp(exponent)
        return float(max(value, 0.0)), f"{species}: estimated Clausius-Clapeyron screening vapor pressure"
    return None, f"{species}: no defensible bundled vapor-pressure model; condensation disabled"


def partition_surface_reservoirs(*, planet: PlanetPhysicalState, temperature_k: float,
                                 atmospheric_mole_fractions: Mapping[str, float],
                                 surface: SurfaceReservoirs,
                                 database: ChemicalDatabase = BUILTIN_DATABASE) -> PhaseReservoirResult:
    area = 4.0 * np.pi * planet.radius_m**2
    total_column_mass = planet.surface_pressure_pa / planet.gravity_m_s2 * area
    mean_mm = sum(database.get(k).molar_mass_kg_mol * x for k, x in atmospheric_mole_fractions.items())
    atmospheric: dict[str, float] = {}
    liquid: dict[str, float] = {}
    solid: dict[str, float] = {}
    volume: dict[str, float] = {}
    fallbacks: list[str] = []
    initial_total = sum(surface.species_mass_kg.values())
    for key, supplied in surface.species_mass_kg.items():
        sp = database.get(key)
        psat, fallback = saturation_pressure_pa(key, temperature_k)
        if fallback:
            fallbacks.append(fallback)
        if psat is None:
            atmospheric[key] = 0.0
            liquid[key] = 0.0
            solid[key] = float(supplied)
            continue
        # Convert maximum equilibrium partial-pressure share to a mass share.
        xmax = min(max(psat / planet.surface_pressure_pa, 0.0), 1.0)
        capacity = total_column_mass * xmax * sp.molar_mass_kg_mol / max(mean_mm, 1e-30)
        vapor = min(float(supplied), capacity)
        condensed = float(supplied) - vapor
        atmospheric[key] = vapor
        is_solid = sp.freezing_point_k is not None and temperature_k < sp.freezing_point_k
        solid[key] = condensed if is_solid else 0.0
        liquid[key] = 0.0 if is_solid else condensed
        if not is_solid and sp.liquid_density_kg_m3:
            volume[key] = condensed / sp.liquid_density_kg_m3
    final_total = sum(atmospheric.values()) + sum(liquid.values()) + sum(solid.values())
    closure = abs(final_total - initial_total) / max(initial_total, 1.0)
    return PhaseReservoirResult(atmospheric, liquid, solid, volume, 0.0, float(closure), tuple(fallbacks))
