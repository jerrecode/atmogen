from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping

import numpy as np

from .chemistry import IdealGibbsEquilibrium, normalized_initial_composition
from .database import BUILTIN_DATABASE, ChemicalDatabase, canonical_species
from .hydrostatic import (
    logarithmic_cell_mean_pressure,
    logarithmic_pressure_interfaces,
    solve_isothermal_hydrostatic,
    solve_temperature_profile_hydrostatic,
)
from .models import (
    CloudResult,
    ColumnBatchInput,
    ColumnBatchDiagnostics,
    ColumnBatchResult,
    ColumnInput,
    ConvergenceReport,
    ElementInventory,
    EnergyBudget,
    Fidelity,
    PhaseReservoirResult,
    PlanetChemistryResult,
    PlanetPhysicalState,
    SolverSettings,
    SpectralResult,
    StellarSpectrum,
    SurfaceReservoirs,
    VerticalProcessResult,
)
from .phase import (
    atmospheric_composition_with_surface_vapor,
    partition_surface_reservoirs,
    saturation_pressure_pa,
)
from .radiation import (
    SIGMA_SB,
    beer_lambert_transmission,
    longwave_optical_depth,
    planck_radiance_w_m2_sr_m,
    rayleigh_optical_depth,
    spectrum_to_srgb,
)
from .thermal import (
    dilute_saturated_log_pressure_gradient,
    dry_adiabatic_log_pressure_gradient,
    solve_dilute_saturated_radiative_convective_profile,
    solve_dry_radiative_convective_profile,
)
from .vertical_processes import solve_vertical_processes
from .version import API_SCHEMA_VERSION, DATA_SCHEMA_VERSION, __version__


def _composition(
    inventory: ElementInventory,
    settings: SolverSettings,
    temperature_k: float,
    pressure_pa: float,
    database: ChemicalDatabase,
) -> tuple[dict[str, float], Mapping[str, object]]:
    if settings.chemistry_mode == "fixed_species":
        fractions = normalized_initial_composition(inventory.initial_species_moles)
        for key in fractions:
            database.get(key)
        return fractions, {
            "backend": "fixed initial molecular state",
            "equilibrated": False,
            "element_relative_residual": 0.0,
            "semantics": inventory.semantics,
        }
    eq = IdealGibbsEquilibrium(database).solve(
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        element_moles=inventory.element_moles,
        initial_species_moles=inventory.initial_species_moles,
    )
    if not eq.converged or not eq.gas_mole_fractions:
        if settings.allow_fidelity_fallback and inventory.initial_species_moles:
            fractions = normalized_initial_composition(inventory.initial_species_moles)
            return fractions, {
                "backend": eq.backend,
                "equilibrated": False,
                "fallback": "fixed_species",
                "failure": eq.message,
                "element_relative_residual": eq.element_relative_residual,
            }
        raise RuntimeError(
            f"chemical equilibrium failed: {eq.message}; "
            f"element residual={eq.element_relative_residual:g}"
        )
    return dict(eq.gas_mole_fractions), {
        "backend": eq.backend,
        "equilibrated": True,
        "element_relative_residual": eq.element_relative_residual,
        "active_species": eq.active_species,
        "pruned_species": eq.pruned_species,
    }


def _interpolate_star(star: StellarSpectrum, wave: np.ndarray) -> np.ndarray:
    return np.interp(
        wave, star.wavelength_m, star.flux_w_m2_m, left=0.0, right=0.0
    )


def _bulk_cloud_tau(
    *,
    condensed_mass_kg: float,
    planet: PlanetPhysicalState,
    settings: SolverSettings,
) -> tuple[float, float]:
    area = 4.0 * np.pi * planet.radius_m**2
    cloud_column = (
        min(
            condensed_mass_kg / area,
            settings.cloud_condensate_column_cap_kg_m2,
        )
        * settings.cloud_suspended_fraction
    )
    density = settings.cloud_particle_density_kg_m3 or 1000.0
    radius = settings.cloud_particle_median_radius_m
    tau = float(3.0 * cloud_column / max(2.0 * density * radius, 1e-300))
    return cloud_column, tau


def _resolved_cloud_tau(
    vertical: VerticalProcessResult, visible_wave: np.ndarray
) -> np.ndarray | None:
    if (
        vertical.cloud_extinction_m_inv.size == 0
        or not np.any(vertical.cloud_extinction_m_inv > 0)
    ):
        return None
    tau_native = np.sum(
        vertical.cloud_extinction_m_inv * vertical.layer_thickness_m[:, None],
        axis=0,
    )
    return np.interp(
        visible_wave,
        vertical.optical_wavelength_m,
        tau_native,
        left=float(tau_native[0]),
        right=float(tau_native[-1]),
    )


def _longwave_tau(
    *,
    planet: PlanetPhysicalState,
    chemistry_composition: Mapping[str, float],
    phase: PhaseReservoirResult,
    area_m2: float,
    database: ChemicalDatabase,
) -> float:
    return longwave_optical_depth(
        surface_pressure_pa=planet.surface_pressure_pa,
        gravity_m_s2=planet.gravity_m_s2,
        mole_fractions=chemistry_composition,
        additional_species_column_kg_m2={
            key: 0.35 * mass / area_m2
            for key, mass in phase.atmospheric_mass_kg.items()
        },
        database=database,
    )


def _resolved_temperature_profile_mode(settings: SolverSettings) -> str:
    requested = settings.temperature_profile_mode
    if requested != "auto":
        return requested
    if settings.fidelity is Fidelity.FAST:
        return "isothermal"
    if settings.fidelity is Fidelity.STANDARD:
        return "dry_radiative_convective"
    return "dilute_saturated"


def _is_isothermal_profile(model: str) -> bool:
    return str(model).startswith("isothermal")


def _select_moist_condensible(
    *,
    planet: PlanetPhysicalState,
    settings: SolverSettings,
    surface_temperature_k: float,
    mole_fractions: Mapping[str, float],
    phase: PhaseReservoirResult,
    database: ChemicalDatabase,
) -> tuple[str | None, tuple[str, ...]]:
    """Select one physically eligible saturated condensable for the dilute backend."""
    condensed = {
        canonical_species(key): float(
            phase.liquid_mass_kg.get(key, 0.0)
            + phase.solid_mass_kg.get(key, 0.0)
        )
        for key in set(phase.liquid_mass_kg) | set(phase.solid_mass_kg)
    }
    condensed = {key: mass for key, mass in condensed.items() if mass > 0}
    if settings.moist_condensible != "auto":
        requested = canonical_species(settings.moist_condensible)
        candidates = (requested,)
    else:
        candidates = tuple(
            key
            for key, _mass in sorted(
                condensed.items(), key=lambda item: (-item[1], item[0])
            )
        )

    notes: list[str] = []
    for key in candidates:
        if key not in condensed:
            notes.append(
                f"{key}: no condensed reservoir is present; saturated convective "
                "constraint not activated"
            )
            continue
        species = database.get(key)
        if species.latent_heat_j_kg is None or species.latent_heat_j_kg <= 0:
            notes.append(
                f"{key}: latent heat unavailable; saturated convective constraint "
                "not activated"
            )
            continue
        if (
            species.freezing_point_k is not None
            and surface_temperature_k < species.freezing_point_k
        ):
            notes.append(
                f"{key}: surface condensate is solid and the database does not "
                "separate sublimation from vaporization latent heat; dry constraint used"
            )
            continue
        carrier_fraction = sum(
            float(value)
            for raw_key, value in mole_fractions.items()
            if canonical_species(raw_key) != key and float(value) > 0
        )
        if carrier_fraction <= 0:
            notes.append(
                f"{key}: no non-condensable carrier atmosphere exists; dilute "
                "saturated approximation is invalid"
            )
            continue
        saturation_pressure, saturation_note = saturation_pressure_pa(
            key, surface_temperature_k
        )
        if saturation_pressure is None:
            notes.append(
                saturation_note
                or f"{key}: saturation-pressure data unavailable; dry constraint used"
            )
            continue
        if (
            saturation_note
            and "estimated" in saturation_note.lower()
            and not settings.moist_allow_estimated_saturation
        ):
            notes.append(
                f"{key}: only an estimated saturation-pressure relation is bundled; "
                "automatic saturated convection requires sourced saturation data"
            )
            continue
        relative_saturation = (
            float(mole_fractions.get(key, 0.0))
            * planet.surface_pressure_pa
            / max(float(saturation_pressure), 1e-300)
        )
        if relative_saturation < settings.moist_saturation_threshold:
            notes.append(
                f"{key}: lower-boundary relative saturation {relative_saturation:.6g} "
                f"is below threshold {settings.moist_saturation_threshold:.6g}"
            )
            continue
        gradient, _r_s, gradient_note = dilute_saturated_log_pressure_gradient(
            pressure_pa=planet.surface_pressure_pa,
            temperature_k=surface_temperature_k,
            mole_fractions=mole_fractions,
            condensible=key,
            max_saturation_mixing_ratio=settings.moist_max_saturation_mixing_ratio,
            allow_estimated_saturation=settings.moist_allow_estimated_saturation,
            database=database,
        )
        if gradient_note and gradient_note not in notes:
            notes.append(gradient_note)
        if gradient is None:
            continue
        return key, tuple(notes)

    return None, tuple(notes)


def _build_atmospheric_profile(
    *,
    planet: PlanetPhysicalState,
    settings: SolverSettings,
    surface_temperature_k: float,
    mole_fractions: Mapping[str, float],
    phase: PhaseReservoirResult,
    longwave_optical_depth_surface: float,
    database: ChemicalDatabase,
):
    """Build the configured vertical thermal/hydrostatic profile and diagnostics."""
    mode = _resolved_temperature_profile_mode(settings)
    empty = np.zeros(settings.resolved_layers, dtype=bool)
    if mode == "isothermal":
        profile = solve_isothermal_hydrostatic(
            surface_pressure_pa=planet.surface_pressure_pa,
            top_pressure_pa=settings.top_pressure_pa,
            temperature_k=surface_temperature_k,
            gravity_m_s2=planet.gravity_m_s2,
            mole_fractions=mole_fractions,
            layers=settings.resolved_layers,
            database=database,
        )
        model = (
            "isothermal_fast"
            if settings.fidelity is Fidelity.FAST
            and settings.temperature_profile_mode == "auto"
            else "isothermal"
        )
        return profile, empty, empty.copy(), model, None, ()

    pressure_interfaces = logarithmic_pressure_interfaces(
        planet.surface_pressure_pa,
        settings.top_pressure_pa,
        settings.resolved_layers,
    )
    pressure = logarithmic_cell_mean_pressure(pressure_interfaces)
    moist_condensible: str | None = None
    thermal_notes: tuple[str, ...] = ()
    saturated_used = empty.copy()

    if mode == "dilute_saturated":
        moist_condensible, selection_notes = _select_moist_condensible(
            planet=planet,
            settings=settings,
            surface_temperature_k=surface_temperature_k,
            mole_fractions=mole_fractions,
            phase=phase,
            database=database,
        )
        thermal_notes = selection_notes
        if moist_condensible is not None:
            (
                temperature_profile,
                adjusted,
                saturated_used,
                adjustment_notes,
            ) = solve_dilute_saturated_radiative_convective_profile(
                pressure_pa=pressure,
                surface_pressure_pa=planet.surface_pressure_pa,
                surface_temperature_k=surface_temperature_k,
                longwave_optical_depth_surface=longwave_optical_depth_surface,
                mole_fractions=mole_fractions,
                condensible=moist_condensible,
                max_saturation_mixing_ratio=(
                    settings.moist_max_saturation_mixing_ratio
                ),
                allow_estimated_saturation=(
                    settings.moist_allow_estimated_saturation
                ),
                optical_depth_pressure_exponent=(
                    settings.gray_optical_depth_pressure_exponent
                ),
                database=database,
            )
            thermal_notes = tuple(
                dict.fromkeys((*thermal_notes, *adjustment_notes))
            )
            actual_mode = "dilute_saturated_gray_radiative_convective"
        else:
            if (
                settings.temperature_profile_mode == "dilute_saturated"
                and not settings.allow_fidelity_fallback
            ):
                raise RuntimeError(
                    "explicit dilute_saturated temperature profile is not physically "
                    "eligible and allow_fidelity_fallback is false: "
                    + "; ".join(thermal_notes)
                )
            temperature_profile, adjusted = solve_dry_radiative_convective_profile(
                pressure_pa=pressure,
                surface_pressure_pa=planet.surface_pressure_pa,
                surface_temperature_k=surface_temperature_k,
                longwave_optical_depth_surface=longwave_optical_depth_surface,
                mole_fractions=mole_fractions,
                optical_depth_pressure_exponent=(
                    settings.gray_optical_depth_pressure_exponent
                ),
                database=database,
            )
            actual_mode = "dry_gray_radiative_convective"
    else:
        temperature_profile, adjusted = solve_dry_radiative_convective_profile(
            pressure_pa=pressure,
            surface_pressure_pa=planet.surface_pressure_pa,
            surface_temperature_k=surface_temperature_k,
            longwave_optical_depth_surface=longwave_optical_depth_surface,
            mole_fractions=mole_fractions,
            optical_depth_pressure_exponent=(
                settings.gray_optical_depth_pressure_exponent
            ),
            database=database,
        )
        actual_mode = "dry_gray_radiative_convective"

    profile = solve_temperature_profile_hydrostatic(
        pressure_interface_pa=pressure_interfaces,
        temperature_k=temperature_profile,
        gravity_m_s2=planet.gravity_m_s2,
        mole_fractions=mole_fractions,
        database=database,
    )
    return (
        profile,
        adjusted,
        saturated_used,
        actual_mode,
        moist_condensible,
        thermal_notes,
    )


def solve_planet(
    *,
    planet: PlanetPhysicalState,
    star: StellarSpectrum,
    inventory: ElementInventory,
    surface: SurfaceReservoirs | None = None,
    settings: SolverSettings | None = None,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> PlanetChemistryResult:
    """Solve a deterministic representative vertical column and global reservoirs."""
    cfg = settings or SolverSettings()
    reservoirs = surface or SurfaceReservoirs()
    if cfg.top_pressure_pa >= planet.surface_pressure_pa:
        raise ValueError("top pressure must be below surface pressure")

    temperature = planet.initial_surface_temperature_k
    previous_composition: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    chemistry_meta: Mapping[str, object] = {}
    energy_imbalance = float("inf")
    relative_t = float("inf")
    composition_delta = float("inf")
    phase: PhaseReservoirResult | None = None
    cloud = CloudResult(
        {}, cfg.cloud_particle_median_radius_m, 0.0, "equilibrium_bulk"
    )
    vertical: VerticalProcessResult | None = None
    chemistry_composition: dict[str, float] = {}
    visible_wave = np.linspace(360e-9, 830e-9, 189)
    cloud_optical_wave = np.linspace(360e-9, 830e-9, 13)
    incident = _interpolate_star(star, visible_wave)
    area = 4.0 * np.pi * planet.radius_m**2
    profile_model = "isothermal_fast"
    convective_adjusted = np.zeros(cfg.resolved_layers, dtype=bool)
    saturated_constraint = np.zeros(cfg.resolved_layers, dtype=bool)
    moist_condensible: str | None = None
    thermal_notes: tuple[str, ...] = ()
    tau_lw = 0.0

    for iteration in range(1, cfg.max_iterations + 1):
        chemistry_composition, chemistry_meta = _composition(
            inventory,
            cfg,
            temperature,
            planet.surface_pressure_pa,
            database,
        )
        phase = partition_surface_reservoirs(
            planet=planet,
            temperature_k=temperature,
            atmospheric_mole_fractions=chemistry_composition,
            surface=reservoirs,
            database=database,
            activity_model=cfg.activity_model,
            liquid_phase_split=cfg.liquid_phase_split,
        )
        composition, _surface_source = atmospheric_composition_with_surface_vapor(
            planet=planet,
            atmospheric_mole_fractions=chemistry_composition,
            surface_vapor_mass_kg=phase.atmospheric_mass_kg,
            database=database,
        )
        composition_delta = (
            max(
                abs(
                    composition.get(k, 0.0)
                    - previous_composition.get(k, 0.0)
                )
                for k in set(composition) | set(previous_composition)
            )
            if previous_composition is not None
            else 1.0
        )

        tau_lw = _longwave_tau(
            planet=planet,
            chemistry_composition=chemistry_composition,
            phase=phase,
            area_m2=area,
            database=database,
        )
        (
            profile_iter,
            convective_adjusted,
            saturated_constraint,
            profile_model,
            moist_condensible,
            thermal_notes,
        ) = _build_atmospheric_profile(
            planet=planet,
            settings=cfg,
            surface_temperature_k=temperature,
            mole_fractions=composition,
            phase=phase,
            longwave_optical_depth_surface=tau_lw,
            database=database,
        )
        vertical = solve_vertical_processes(
            profile=profile_iter,
            phase=phase,
            planet=planet,
            settings=cfg,
            optical_wavelength_m=cloud_optical_wave,
            database=database,
        )

        condensed = sum(phase.liquid_mass_kg.values()) + sum(
            phase.solid_mass_kg.values()
        )
        _cloud_column, cloud_tau_bulk = _bulk_cloud_tau(
            condensed_mass_kg=condensed,
            planet=planet,
            settings=cfg,
        )
        cloud_tau_spectral = (
            _resolved_cloud_tau(vertical, visible_wave)
            if cfg.cloud_mode == "lognormal_sedimentation"
            else None
        )
        if cloud_tau_spectral is not None:
            cloud_tau_for_radiation: np.ndarray | float = cloud_tau_spectral
            cloud_tau_visible = float(
                np.interp(550e-9, visible_wave, cloud_tau_spectral)
            )
            cloud_model = (
                "resolved lognormal Mie extinction + sedimentation/precipitation"
            )
        else:
            cloud_tau_for_radiation = cloud_tau_bulk
            cloud_tau_visible = cloud_tau_bulk
            cloud_model = (
                "resolved lognormal sedimentation with bulk-gray optical fallback"
                if cfg.cloud_mode == "lognormal_sedimentation"
                else "equilibrium bulk condensate; bounded suspended fraction"
            )
        sigma_ln = np.log(cfg.cloud_particle_geometric_std)
        effective_radius = cfg.cloud_particle_median_radius_m * np.exp(
            2.5 * sigma_ln**2
        )
        cloud = CloudResult(
            {
                k: phase.liquid_mass_kg.get(k, 0.0)
                + phase.solid_mass_kg.get(k, 0.0)
                for k in set(phase.liquid_mass_kg) | set(phase.solid_mass_kg)
            },
            float(effective_radius),
            float(cloud_tau_visible),
            cloud_model,
        )

        tau_r = rayleigh_optical_depth(
            wavelength_m=visible_wave,
            surface_pressure_pa=planet.surface_pressure_pa,
            gravity_m_s2=planet.gravity_m_s2,
            mole_fractions=composition,
            database=database,
        )
        tau_total = tau_r + cloud_tau_for_radiation
        trans = beer_lambert_transmission(tau_total)
        rayleigh_reflect = tau_total / (tau_total + 2.0)
        spectral_albedo = np.clip(
            rayleigh_reflect
            + (1.0 - rayleigh_reflect)
            * planet.surface_albedo_initial
            * trans,
            0.0,
            0.98,
        )
        denom = np.trapezoid(incident, visible_wave)
        visible_bond = (
            float(
                np.trapezoid(spectral_albedo * incident, visible_wave) / denom
            )
            if denom > 0
            else planet.surface_albedo_initial
        )
        bond = float(
            np.clip(
                0.72 * visible_bond + 0.28 * planet.surface_albedo_initial,
                0.0,
                0.95,
            )
        )
        incoming_mean = star.bolometric_flux_w_m2 / 4.0
        absorbed = incoming_mean * (1.0 - bond)
        teff = (
            (absorbed + planet.internal_heat_flux_w_m2) / SIGMA_SB
        ) ** 0.25
        target = teff * (1.0 + 0.75 * tau_lw) ** 0.25
        target = float(np.clip(target, 20.0, 4000.0))
        updated = (
            (1.0 - cfg.relaxation) * temperature + cfg.relaxation * target
        )
        relative_t = abs(updated - temperature) / max(temperature, 1.0)
        outgoing = SIGMA_SB * teff**4
        energy_imbalance = (
            absorbed + planet.internal_heat_flux_w_m2 - outgoing
        )
        adjusted_count = float(np.count_nonzero(convective_adjusted))
        history.append(
            {
                "temperature_relative": relative_t,
                "composition_absolute": composition_delta,
                "energy_w_m2": abs(energy_imbalance),
                "cloud_optical_depth_visible": float(cloud_tau_visible),
                "surface_precipitation_kg_m2_step": float(
                    vertical.surface_precipitation_kg_m2
                ),
                "convective_adjusted_layers": adjusted_count,
                "dry_convective_adjusted_layers": adjusted_count,
                "saturated_constraint_layers": float(
                    np.count_nonzero(saturated_constraint)
                ),
                "profile_temperature_range_k": float(
                    np.ptp(profile_iter.temperature_k)
                ),
            }
        )
        temperature = updated
        previous_composition = composition
        if (
            relative_t <= cfg.relative_temperature_tolerance
            and composition_delta <= cfg.composition_tolerance
            and abs(energy_imbalance) <= cfg.energy_tolerance_w_m2
        ):
            break

    if phase is None or vertical is None or previous_composition is None:
        raise RuntimeError("column solve did not produce a physical state")
    converged = (
        relative_t <= cfg.relative_temperature_tolerance
        and composition_delta <= cfg.composition_tolerance
        and abs(energy_imbalance) <= cfg.energy_tolerance_w_m2
    )

    tau_lw = _longwave_tau(
        planet=planet,
        chemistry_composition=chemistry_composition,
        phase=phase,
        area_m2=area,
        database=database,
    )
    (
        profile,
        convective_adjusted,
        saturated_constraint,
        profile_model,
        moist_condensible,
        thermal_notes,
    ) = _build_atmospheric_profile(
        planet=planet,
        settings=cfg,
        surface_temperature_k=temperature,
        mole_fractions=previous_composition,
        phase=phase,
        longwave_optical_depth_surface=tau_lw,
        database=database,
    )
    vertical = solve_vertical_processes(
        profile=profile,
        phase=phase,
        planet=planet,
        settings=cfg,
        optical_wavelength_m=cloud_optical_wave,
        database=database,
    )
    final_cloud_tau = (
        _resolved_cloud_tau(vertical, visible_wave)
        if cfg.cloud_mode == "lognormal_sedimentation"
        else None
    )
    if final_cloud_tau is None:
        final_cloud_tau = np.full(
            visible_wave.shape, cloud.optical_depth_visible, dtype=float
        )
    tau_r = rayleigh_optical_depth(
        wavelength_m=visible_wave,
        surface_pressure_pa=planet.surface_pressure_pa,
        gravity_m_s2=planet.gravity_m_s2,
        mole_fractions=previous_composition,
        database=database,
    )
    tau_total = tau_r + final_cloud_tau
    trans = beer_lambert_transmission(tau_total)
    rayleigh_reflect = tau_total / (tau_total + 2.0)
    spectral_albedo = np.clip(
        rayleigh_reflect
        + (1.0 - rayleigh_reflect) * planet.surface_albedo_initial * trans,
        0.0,
        0.98,
    )
    denom = np.trapezoid(incident, visible_wave)
    visible_bond = (
        float(np.trapezoid(spectral_albedo * incident, visible_wave) / denom)
        if denom > 0
        else planet.surface_albedo_initial
    )
    bond = float(
        np.clip(
            0.72 * visible_bond + 0.28 * planet.surface_albedo_initial,
            0.0,
            0.95,
        )
    )
    incoming_mean = star.bolometric_flux_w_m2 / 4.0
    absorbed = incoming_mean * (1.0 - bond)
    teff = ((absorbed + planet.internal_heat_flux_w_m2) / SIGMA_SB) ** 0.25
    energy_imbalance = (
        absorbed + planet.internal_heat_flux_w_m2 - SIGMA_SB * teff**4
    )
    thermal_wave = np.geomspace(2e-6, 80e-6, 256)
    thermal_shape = np.pi * planck_radiance_w_m2_sr_m(thermal_wave, teff)
    thermal_shape *= (SIGMA_SB * teff**4) / np.trapezoid(
        thermal_shape, thermal_wave
    )
    reflected = incident * spectral_albedo
    spectra = SpectralResult(
        visible_wave,
        incident,
        reflected,
        trans,
        tau_r,
        thermal_wave,
        thermal_shape,
        spectral_albedo,
        bond,
        min(1.5 * bond, 1.0),
        spectrum_to_srgb(visible_wave, reflected),
    )
    budget = EnergyBudget(
        incoming_mean,
        absorbed,
        SIGMA_SB * teff**4,
        planet.internal_heat_flux_w_m2,
        energy_imbalance,
        tau_lw,
    )
    convergence = ConvergenceReport(
        converged,
        iteration,
        relative_t,
        composition_delta,
        energy_imbalance,
        tuple(history),
    )
    total_negative = any(
        v < 0
        for collection in (
            phase.atmospheric_mass_kg,
            phase.liquid_mass_kg,
            phase.solid_mass_kg,
        )
        for v in collection.values()
    )
    dry_gradient = (
        None
        if _is_isothermal_profile(profile_model)
        else dry_adiabatic_log_pressure_gradient(previous_composition, database)
    )
    combined_fallbacks = tuple(
        dict.fromkeys(
            (*phase.fallbacks, *vertical.fallbacks, *thermal_notes)
        )
    )
    adjusted_count = int(np.count_nonzero(convective_adjusted))
    diagnostics = {
        "finite": bool(
            np.isfinite(profile.pressure_pa).all()
            and np.isfinite(profile.temperature_k).all()
            and np.isfinite(profile.density_kg_m3).all()
            and np.isfinite(vertical.cloud_condensate_kg_m2).all()
        ),
        "non_negative": not total_negative,
        "hydrostatic_relative_residual": profile.hydrostatic_relative_residual,
        "reservoir_mass_closure_relative": phase.mass_closure_relative,
        "vertical_process_mass_closure_relative": vertical.mass_closure_relative,
        "surface_precipitation_kg_m2_per_microphysics_step": (
            vertical.surface_precipitation_kg_m2
        ),
        "reevaporation_latent_cooling_w_m2_diagnostic": (
            vertical.reevaporation_latent_cooling_w_m2
        ),
        "chemistry": chemistry_meta,
        "fallbacks": combined_fallbacks,
        "thermal_fallbacks": thermal_notes,
        "liquid_phase_count": len(phase.liquid_phases),
        "activity_model": phase.activity_model,
        "vertical_transport_mode": cfg.vertical_transport_mode,
        "cloud_process_model": vertical.model,
        "temperature_profile_requested_mode": cfg.temperature_profile_mode,
        "temperature_profile_model": profile_model,
        "temperature_profile_range_k": float(np.ptp(profile.temperature_k)),
        "convective_adjusted_layers": adjusted_count,
        "dry_convective_adjusted_layers": adjusted_count,
        "saturated_convective_constraint_layers": int(
            np.count_nonzero(saturated_constraint)
        ),
        "moist_condensible": moist_condensible,
        "dry_adiabatic_log_pressure_gradient": dry_gradient,
        "gray_optical_depth_pressure_exponent": (
            None
            if _is_isothermal_profile(profile_model)
            else cfg.gray_optical_depth_pressure_exponent
        ),
    }
    provenance = {
        "atmogen_version": __version__,
        "api_schema_version": API_SCHEMA_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "database_sha256": database.revision_hash,
        "fidelity": cfg.fidelity.value,
        "chemistry_mode": cfg.chemistry_mode,
        "radiation_mode": cfg.radiation_mode,
        "temperature_profile_mode_requested": cfg.temperature_profile_mode,
        "temperature_profile_model": profile_model,
        "gray_optical_depth_pressure_exponent": (
            cfg.gray_optical_depth_pressure_exponent
        ),
        "moist_condensible_requested": cfg.moist_condensible,
        "moist_condensible_selected": moist_condensible,
        "moist_saturation_threshold": cfg.moist_saturation_threshold,
        "moist_max_saturation_mixing_ratio": (
            cfg.moist_max_saturation_mixing_ratio
        ),
        "moist_allow_estimated_saturation": (
            cfg.moist_allow_estimated_saturation
        ),
        "cloud_mode": cfg.cloud_mode,
        "activity_model_requested": cfg.activity_model,
        "liquid_phase_split": cfg.liquid_phase_split,
        "vertical_transport_mode": cfg.vertical_transport_mode,
        "eddy_diffusivity_m2_s": cfg.eddy_diffusivity_m2_s,
        "cloud_suspended_fraction": cfg.cloud_suspended_fraction,
        "cloud_condensate_column_cap_kg_m2": (
            cfg.cloud_condensate_column_cap_kg_m2
        ),
        "cloud_particle_median_radius_m": cfg.cloud_particle_median_radius_m,
        "cloud_particle_geometric_std": cfg.cloud_particle_geometric_std,
        "cloud_particle_density_kg_m3": cfg.cloud_particle_density_kg_m3,
        "cloud_refractive_index_real": cfg.cloud_refractive_index_real,
        "cloud_refractive_index_imag": cfg.cloud_refractive_index_imag,
        "cloud_microphysics_timestep_s": cfg.cloud_microphysics_timestep_s,
        "cloud_reevaporation_timescale_s": cfg.cloud_reevaporation_timescale_s,
        "stellar_spectrum": star.provenance,
        "condensable_vertical_depletion_factor_fast": 0.35,
        "surface_vapor_band_optical_depth_cap_fast": 1.5,
        "vertical_cloud_initialization": (
            "uniform condensate-to-air mass ratio over a bounded suspended column"
        ),
        "latent_heat_semantics": (
            "saturated profile uses a single-condensable dilute approximate lapse "
            "constraint where eligible; precipitation re-evaporation latent cooling "
            "remains diagnostic redistribution until layerwise energy integration"
        ),
        "limitations": (
            "semi-gray longwave, fixed-pressure reservoir coupling, ideal-volume "
            "liquid density, bulk cold-trap depletion, reduced-order cloud source, "
            "single-condensable dilute saturated adjustment, no solid-phase moist "
            "adiabat without separate sublimation latent heat, and no layerwise "
            "radiative energy-flux convergence remain approximations"
        ),
    }
    return PlanetChemistryResult(
        profile,
        phase,
        cloud,
        vertical,
        spectra,
        budget,
        convergence,
        diagnostics,
        provenance,
    )


def _fingerprint_value(value: object) -> object:
    """Convert typed solver state into a deterministic JSON-compatible payload."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _fingerprint_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _fingerprint_value(value.item())
    if isinstance(value, (tuple, list)):
        return [_fingerprint_value(item) for item in value]
    return value


def column_state_fingerprint(
    column: ColumnInput,
    star: StellarSpectrum,
    settings: SolverSettings | None = None,
    *,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> str:
    """Return the stable identity of a complete host-requested column state.

    The identity includes numerical inputs, provenance-affecting strings, solver/API
    versions and the chemical database revision. It is suitable for request
    coalescing and persistent cache keys; it is not a scientific checksum of an
    already computed result.
    """
    cfg = settings or SolverSettings()
    raw = json.dumps(
        {
            "fingerprint_schema": "atmogen-column-state-v1",
            "atmogen_version": __version__,
            "api_schema_version": API_SCHEMA_VERSION,
            "data_schema_version": DATA_SCHEMA_VERSION,
            "database_sha256": database.revision_hash,
            "column": _fingerprint_value(column),
            "settings": _fingerprint_value(cfg),
            "star": _fingerprint_value(star),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _scaled_stellar_spectrum(
    star: StellarSpectrum, scale: float
) -> StellarSpectrum:
    value = float(scale)
    if value == 1.0:
        return star
    return StellarSpectrum(
        wavelength_m=star.wavelength_m,
        flux_w_m2_m=np.asarray(star.flux_w_m2_m, dtype=float) * value,
        provenance=f"{star.provenance}; column stellar flux scale={value:.12g}",
    )


def solve_columns(
    batch: ColumnBatchInput,
    settings: SolverSettings | None = None,
    *,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[PlanetChemistryResult, ...]:
    """Solve a column batch with exact-state de-duplication and per-column forcing."""
    return solve_columns_with_diagnostics(
        batch, settings, database=database
    ).results


def solve_columns_with_diagnostics(
    batch: ColumnBatchInput,
    settings: SolverSettings | None = None,
    *,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> ColumnBatchResult:
    """Solve an ordered batch and expose stable cache/provenance diagnostics."""
    cfg = settings or SolverSettings()
    cache: dict[str, PlanetChemistryResult] = {}
    output: list[PlanetChemistryResult] = []
    fingerprints: list[str] = []
    reused: list[bool] = []
    for column in batch.columns:
        column_star = _scaled_stellar_spectrum(
            batch.star, column.stellar_flux_scale
        )
        key = column_state_fingerprint(
            column, batch.star, cfg, database=database
        )
        was_reused = key in cache
        if key not in cache:
            cache[key] = solve_planet(
                planet=column.planet,
                star=column_star,
                inventory=column.inventory,
                surface=column.surface,
                settings=cfg,
                database=database,
            )
        output.append(cache[key])
        fingerprints.append(key)
        reused.append(was_reused)

    unique_fingerprints = tuple(sorted(cache))
    unique_index = {key: idx for idx, key in enumerate(unique_fingerprints)}
    fallback_sets = [tuple(result.diagnostics.get("fallbacks", ())) for result in output]
    input_count = len(output)
    unique_count = len(cache)
    diagnostics = ColumnBatchDiagnostics(
        input_count=input_count,
        unique_state_count=unique_count,
        deduplicated_count=input_count - unique_count,
        deduplication_ratio=(
            float((input_count - unique_count) / input_count)
            if input_count
            else 0.0
        ),
        converged_count=sum(result.convergence.converged for result in output),
        fallback_column_count=sum(bool(items) for items in fallback_sets),
        fallback_event_count=sum(len(items) for items in fallback_sets),
        fingerprints=tuple(fingerprints),
        unique_fingerprints=unique_fingerprints,
        unique_state_index=tuple(unique_index[key] for key in fingerprints),
        reused=tuple(reused),
        per_column_provenance=tuple(dict(result.provenance) for result in output),
        database_sha256=database.revision_hash,
    )
    return ColumnBatchResult(tuple(output), diagnostics)
