from __future__ import annotations

from typing import Mapping

import numpy as np

from .database import BUILTIN_DATABASE, ChemicalDatabase, canonical_species
from .liquids import (liquid_mixture_density_kg_m3, liquid_phase_stability,
                      select_activity_model)
from .models import (LiquidPhaseState, PhaseReservoirResult, PlanetPhysicalState,
                     SurfaceReservoirs)


def saturation_pressure_pa(species: str, temperature_k: float) -> tuple[float | None, str | None]:
    """Return equilibrium vapor pressure and provenance/fallback note.

    Water uses the Murphy & Koop (2005) vapor-pressure relation over ice from
    110 K up to the triple point and the IAPWS saturation release form from the
    triple point through the critical point. Other bundled condensables retain
    bounded Clausius-Clapeyron screening fits and are explicitly labelled as
    estimates rather than measured/fitted high-fidelity vapor-pressure models.
    """
    key = canonical_species(species)
    t = float(temperature_k)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("temperature_k must be finite and positive")

    if key == "H2O" and 110.0 <= t < 273.16:
        # Murphy & Koop (2005), QJRMS 131, 1539-1565, eq. 7:
        # ln(p_ice / Pa) = 9.550426 - 5723.265/T + 3.53068 ln(T) - 0.00728332 T.
        ln_pressure = (
            9.550426
            - 5723.265 / t
            + 3.53068 * np.log(t)
            - 0.00728332 * t
        )
        return float(np.exp(ln_pressure)), (
            "H2O: Murphy-Koop (2005) equilibrium vapor pressure over ice"
        )

    if key == "H2O" and 273.16 <= t <= 647.096:
        tc, pc = 647.096, 22.064e6
        theta = 1.0 - t / tc
        a = (
            -7.85951783,
            1.84408259,
            -11.7866497,
            22.6807411,
            -15.9618719,
            1.80122502,
        )
        powers = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
        return float(
            pc
            * np.exp(
                tc
                / t
                * sum(c * theta**p for c, p in zip(a, powers, strict=True))
            )
        ), None

    if key == "H2O":
        return None, (
            "H2O: no bundled equilibrium vapor-pressure relation outside "
            "110-647.096 K; condensation screening disabled"
        )

    anchors = {
        "CO2": (216.58, 5.18e5, 5.74e5, 44.0095e-3),
        "CH4": (90.69, 1.17e4, 5.10e5, 16.0425e-3),
        "NH3": (195.40, 6.06e3, 1.37e6, 17.03052e-3),
        "C2H6": (90.35, 1.1, 4.89e5, 30.0690e-3),
        "SO2": (197.67, 1.67e4, 3.89e5, 64.066e-3),
    }
    if key in anchors:
        tref, pref, latent, mm = anchors[key]
        exponent = np.clip(
            -latent
            * mm
            / 8.31446261815324
            * (1.0 / t - 1.0 / tref),
            -745.0,
            700.0,
        )
        value = pref * np.exp(exponent)
        return float(max(value, 0.0)), (
            f"{key}: estimated Clausius-Clapeyron screening vapor pressure"
        )
    return None, (
        f"{key}: no defensible bundled vapor-pressure model; condensation disabled"
    )


def _normalised_background(mole_fractions: Mapping[str, float],
                           database: ChemicalDatabase) -> dict[str, float]:
    combined: dict[str, float] = {}
    for raw_key, value in mole_fractions.items():
        amount = float(value)
        if amount <= 0:
            continue
        key = canonical_species(raw_key)
        database.get(key)
        combined[key] = combined.get(key, 0.0) + amount
    total = sum(combined.values())
    if total <= 0:
        raise ValueError("atmospheric_mole_fractions must contain a positive amount")
    return {key: value / total for key, value in combined.items()}


def atmospheric_composition_with_surface_vapor(
    *,
    planet: PlanetPhysicalState,
    atmospheric_mole_fractions: Mapping[str, float],
    surface_vapor_mass_kg: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[dict[str, float], dict[str, float]]:
    """Combine a fixed-pressure background atmosphere with tagged surface vapor.

    ``planet.surface_pressure_pa`` remains the bulk boundary condition. Surface
    vapor therefore occupies part of the fixed atmospheric column and the dry
    background inventory is reduced by the same mass. The second return value is
    the mole-fraction contribution attributable specifically to the surface
    reservoir, which preserves source bookkeeping when a molecule also exists in
    the background composition.
    """
    background = _normalised_background(atmospheric_mole_fractions, database)
    area = 4.0 * np.pi * planet.radius_m**2
    atmospheric_mass = planet.surface_pressure_pa / planet.gravity_m_s2 * area
    dry_mean_mm = sum(database.get(key).molar_mass_kg_mol * value
                      for key, value in background.items())
    vapor = {
        canonical_species(key): max(float(value), 0.0)
        for key, value in surface_vapor_mass_kg.items()
        if float(value) > 0
    }
    vapor_total = sum(vapor.values())
    if vapor_total > atmospheric_mass * (1.0 + 2.0e-12):
        raise ValueError("surface vapor mass exceeds the prescribed fixed-pressure atmospheric column")
    dry_mass = max(atmospheric_mass - vapor_total, 0.0)
    dry_total_moles = dry_mass / max(dry_mean_mm, 1.0e-30)
    species_moles = {
        key: dry_total_moles * value for key, value in background.items()
    }
    source_moles: dict[str, float] = {}
    for key, mass in vapor.items():
        amount = mass / database.get(key).molar_mass_kg_mol
        source_moles[key] = amount
        species_moles[key] = species_moles.get(key, 0.0) + amount
    total_moles = sum(species_moles.values())
    if total_moles <= 0:
        raise RuntimeError("combined atmosphere contains no material")
    composition = {key: value / total_moles for key, value in species_moles.items()
                   if value > 0}
    source_fraction = {key: value / total_moles for key, value in source_moles.items()
                       if value > 0}
    return composition, source_fraction


def _target_vapor_masses(
    *,
    planet: PlanetPhysicalState,
    background: Mapping[str, float],
    target_surface_mole_fractions: Mapping[str, float],
    database: ChemicalDatabase,
) -> dict[str, float]:
    """Convert desired surface-vapor partial-pressure shares to fixed-column masses."""
    area = 4.0 * np.pi * planet.radius_m**2
    atmospheric_mass = planet.surface_pressure_pa / planet.gravity_m_s2 * area
    dry_mean_mm = sum(database.get(key).molar_mass_kg_mol * value
                      for key, value in background.items())
    vapor_share = float(sum(target_surface_mole_fractions.values()))
    vapor_share = float(np.clip(vapor_share, 0.0, 1.0))
    mean_mm = (1.0 - vapor_share) * dry_mean_mm
    mean_mm += sum(float(value) * database.get(key).molar_mass_kg_mol
                   for key, value in target_surface_mole_fractions.items())
    mean_mm = max(mean_mm, 1.0e-30)
    return {
        key: atmospheric_mass * float(value) * database.get(key).molar_mass_kg_mol / mean_mm
        for key, value in target_surface_mole_fractions.items()
    }


def partition_surface_reservoirs(
    *,
    planet: PlanetPhysicalState,
    temperature_k: float,
    atmospheric_mole_fractions: Mapping[str, float],
    surface: SurfaceReservoirs,
    database: ChemicalDatabase = BUILTIN_DATABASE,
    activity_model: str = "auto",
    liquid_phase_split: bool = True,
) -> PhaseReservoirResult:
    """Partition finite surface inventories across vapor, liquid, and solid phases.

    Liquid vapor pressure follows ``p_i = x_i gamma_i P_sat,i`` when more than one
    liquid component is present. ``auto`` uses NRTL only when the database has a
    complete directed parameter set for the active liquid components; otherwise it
    records and uses the ideal-solution fallback. The planet pressure is presently
    a fixed boundary rather than a solved reservoir variable.
    """
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    background = _normalised_background(atmospheric_mole_fractions, database)
    supplied: dict[str, float] = {}
    for raw_key, value in surface.species_mass_kg.items():
        key = canonical_species(raw_key)
        database.get(key)
        supplied[key] = supplied.get(key, 0.0) + float(value)
    area = 4.0 * np.pi * planet.radius_m**2
    atmospheric_column_mass = planet.surface_pressure_pa / planet.gravity_m_s2 * area
    initial_total = sum(supplied.values())
    fallbacks: list[str] = []

    def note(message: str | None) -> None:
        if message and message not in fallbacks:
            fallbacks.append(message)

    psat: dict[str, float | None] = {}
    is_solid: dict[str, bool] = {}
    for key in supplied:
        pressure, fallback = saturation_pressure_pa(key, temperature_k)
        psat[key] = pressure
        note(fallback)
        species = database.get(key)
        is_solid[key] = bool(
            species.freezing_point_k is not None and temperature_k < species.freezing_point_k
        )

    vapor = {key: 0.0 for key in supplied}
    selected_name = "ideal"
    for _ in range(80):
        remaining = {key: max(supplied[key] - vapor[key], 0.0) for key in supplied}
        liquid_keys = tuple(
            key for key in sorted(supplied)
            if psat[key] is not None and not is_solid[key]
            and remaining[key] > max(1.0, supplied[key] * 1.0e-13)
        )
        solid_keys = tuple(
            key for key in sorted(supplied)
            if psat[key] is not None and is_solid[key]
            and remaining[key] > max(1.0, supplied[key] * 1.0e-13)
        )
        target_y: dict[str, float] = {}
        if liquid_keys:
            liquid_moles = {
                key: remaining[key] / database.get(key).molar_mass_kg_mol
                for key in liquid_keys
            }
            total_liquid_moles = sum(liquid_moles.values())
            x = {key: value / total_liquid_moles for key, value in liquid_moles.items()}
            model, model_fallbacks = select_activity_model(
                species=liquid_keys, database=database, mode=activity_model
            )
            selected_name = model.name
            for fallback in model_fallbacks:
                note(fallback)
            gamma = model.activity_coefficients(
                temperature_k=temperature_k, mole_fractions=x
            )
            for key in liquid_keys:
                target_y[key] = max(
                    float(x[key]) * float(gamma[key]) * float(psat[key])
                    / planet.surface_pressure_pa,
                    0.0,
                )
        for key in solid_keys:
            # Water now uses the sourced Murphy-Koop ice relation here. Other
            # bundled species retain explicitly labelled screening sublimation fits.
            target_y[key] = max(float(psat[key]) / planet.surface_pressure_pa, 0.0)

        target_total = sum(target_y.values())
        if target_total > 1.0:
            note(
                "equilibrium condensable vapor pressure exceeded prescribed total pressure; "
                "normalized surface-vapor shares at the fixed-pressure boundary"
            )
            target_y = {key: value / target_total for key, value in target_y.items()}
        capacities = _target_vapor_masses(
            planet=planet, background=background,
            target_surface_mole_fractions=target_y, database=database,
        )
        new_vapor: dict[str, float] = {}
        for key, total_mass in supplied.items():
            if key in capacities:
                new_vapor[key] = min(total_mass, max(capacities[key], 0.0))
            elif psat[key] is None:
                new_vapor[key] = 0.0
            else:
                # An exhausted finite reservoir has no condensed phase left to
                # impose saturation; keep its undersaturated gas inventory.
                new_vapor[key] = min(total_mass, vapor[key])

        total_vapor = sum(new_vapor.values())
        if total_vapor > atmospheric_column_mass:
            note(
                "finite surface vapor exceeded prescribed atmospheric column mass; "
                "scaled vapor inventory to the fixed-pressure column"
            )
            scale = atmospheric_column_mass / max(total_vapor, 1.0e-30)
            new_vapor = {key: value * scale for key, value in new_vapor.items()}

        change = max(
            (abs(new_vapor[key] - vapor[key]) / max(supplied[key], 1.0))
            for key in supplied
        ) if supplied else 0.0
        vapor = new_vapor
        if change <= 2.0e-11:
            break
    else:
        note("surface vapor/liquid partition reached iteration limit")

    liquid: dict[str, float] = {}
    solid: dict[str, float] = {}
    volume: dict[str, float] = {}
    for key, total_mass in supplied.items():
        condensed = max(total_mass - vapor[key], 0.0)
        if psat[key] is None:
            # Preserve the previous conservative behavior: a species for which no
            # vapor model exists is retained in a condensed reservoir rather than
            # silently transformed into an unconstrained gas.
            solid[key] = condensed
            liquid[key] = 0.0
        elif is_solid[key]:
            solid[key] = condensed
            liquid[key] = 0.0
        else:
            solid[key] = 0.0
            liquid[key] = condensed
            density = database.get(key).liquid_density_kg_m3
            if density is not None and density > 0:
                volume[key] = condensed / density

    combined_atmosphere, source_y = atmospheric_composition_with_surface_vapor(
        planet=planet, atmospheric_mole_fractions=background,
        surface_vapor_mass_kg=vapor, database=database,
    )

    liquid_keys = tuple(key for key in sorted(liquid) if liquid[key] > 0)
    liquid_phases: list[LiquidPhaseState] = []
    final_activity_name = "ideal"
    if liquid_keys:
        liquid_moles = {
            key: liquid[key] / database.get(key).molar_mass_kg_mol
            for key in liquid_keys
        }
        total_liquid_moles = sum(liquid_moles.values())
        overall_x = {key: value / total_liquid_moles for key, value in liquid_moles.items()}
        model, model_fallbacks = select_activity_model(
            species=liquid_keys, database=database, mode=activity_model
        )
        final_activity_name = model.name
        for fallback in model_fallbacks:
            note(fallback)
        if liquid_phase_split and len(liquid_keys) > 1:
            split = liquid_phase_stability(
                temperature_k=temperature_k, mole_fractions=overall_x,
                activity_model=model,
            )
            phase_fractions = split.phase_fractions_mol
            phase_compositions = split.phase_compositions
            if not split.single_phase_stable:
                note(f"liquid-liquid phase separation selected by {split.method}")
        else:
            phase_fractions = (1.0,)
            phase_compositions = (overall_x,)
        for fraction, composition in zip(phase_fractions, phase_compositions, strict=True):
            species_moles = {
                key: float(fraction) * total_liquid_moles * float(composition.get(key, 0.0))
                for key in liquid_keys
                if float(composition.get(key, 0.0)) > 0
            }
            species_mass = {
                key: amount * database.get(key).molar_mass_kg_mol
                for key, amount in species_moles.items()
            }
            density, missing_density = liquid_mixture_density_kg_m3(
                species_mass_kg=species_mass, database=database
            )
            if missing_density:
                note(
                    "liquid mixture density unavailable for "
                    + ", ".join(missing_density)
                )
            phase_mass = sum(species_mass.values())
            phase_volume = phase_mass / density if density is not None and density > 0 else None
            gamma = model.activity_coefficients(
                temperature_k=temperature_k, mole_fractions=composition
            )
            liquid_phases.append(
                LiquidPhaseState(
                    phase_fraction_mol=float(fraction),
                    species_moles=species_moles,
                    mole_fractions=dict(composition),
                    species_mass_kg=species_mass,
                    density_kg_m3=density,
                    volume_m3=phase_volume,
                    activity_coefficients=gamma,
                    activity_model=model.name,
                )
            )
        note(
            "liquid density currently uses ideal volume additivity even when "
            "non-ideal activities are enabled"
        )

    final_total = sum(vapor.values()) + sum(liquid.values()) + sum(solid.values())
    closure = abs(final_total - initial_total) / max(initial_total, 1.0)
    if closure > 2.0e-11:
        raise RuntimeError(f"surface reservoir partition failed mass closure: {closure:g}")
    if combined_atmosphere and abs(sum(combined_atmosphere.values()) - 1.0) > 2.0e-12:
        raise RuntimeError("combined atmospheric mole fractions failed normalization")
    return PhaseReservoirResult(
        atmospheric_mass_kg=vapor,
        liquid_mass_kg=liquid,
        solid_mass_kg=solid,
        liquid_volume_m3=volume,
        latent_heat_flux_w_m2=0.0,
        mass_closure_relative=float(closure),
        fallbacks=tuple(fallbacks),
        surface_vapor_mole_fractions=source_y,
        liquid_phases=tuple(liquid_phases),
        activity_model=final_activity_name if liquid_keys else selected_name,
    )
