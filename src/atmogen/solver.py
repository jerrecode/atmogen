from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Mapping

import numpy as np

from .chemistry import IdealGibbsEquilibrium, normalized_initial_composition
from .database import BUILTIN_DATABASE, ChemicalDatabase
from .hydrostatic import solve_isothermal_hydrostatic
from .models import (CloudResult, ColumnBatchInput, ConvergenceReport, ElementInventory,
                     EnergyBudget, PhaseReservoirResult, PlanetChemistryResult,
                     PlanetPhysicalState, SolverSettings, SpectralResult, StellarSpectrum,
                     SurfaceReservoirs, VerticalProcessResult)
from .phase import (atmospheric_composition_with_surface_vapor,
                    partition_surface_reservoirs)
from .radiation import (SIGMA_SB, beer_lambert_transmission, longwave_optical_depth,
                        planck_radiance_w_m2_sr_m, rayleigh_optical_depth, spectrum_to_srgb)
from .vertical_processes import solve_vertical_processes
from .version import API_SCHEMA_VERSION, DATA_SCHEMA_VERSION, __version__


def _composition(inventory: ElementInventory, settings: SolverSettings, temperature_k: float,
                 pressure_pa: float, database: ChemicalDatabase) -> tuple[dict[str, float], Mapping[str, object]]:
    if settings.chemistry_mode == "fixed_species":
        fractions = normalized_initial_composition(inventory.initial_species_moles)
        for key in fractions:
            database.get(key)
        return fractions, {"backend": "fixed initial molecular state", "equilibrated": False,
                           "element_relative_residual": 0.0, "semantics": inventory.semantics}
    eq = IdealGibbsEquilibrium(database).solve(temperature_k=temperature_k, pressure_pa=pressure_pa,
                                               element_moles=inventory.element_moles,
                                               initial_species_moles=inventory.initial_species_moles)
    if not eq.converged or not eq.gas_mole_fractions:
        if settings.allow_fidelity_fallback and inventory.initial_species_moles:
            fractions = normalized_initial_composition(inventory.initial_species_moles)
            return fractions, {"backend": eq.backend, "equilibrated": False, "fallback": "fixed_species",
                               "failure": eq.message, "element_relative_residual": eq.element_relative_residual}
        raise RuntimeError(f"chemical equilibrium failed: {eq.message}; element residual={eq.element_relative_residual:g}")
    return dict(eq.gas_mole_fractions), {"backend": eq.backend, "equilibrated": True,
                                         "element_relative_residual": eq.element_relative_residual,
                                         "active_species": eq.active_species, "pruned_species": eq.pruned_species}


def _interpolate_star(star: StellarSpectrum, wave: np.ndarray) -> np.ndarray:
    return np.interp(wave, star.wavelength_m, star.flux_w_m2_m, left=0.0, right=0.0)


def _bulk_cloud_tau(*, condensed_mass_kg: float, planet: PlanetPhysicalState,
                    settings: SolverSettings) -> tuple[float, float]:
    area = 4.0 * np.pi * planet.radius_m**2
    cloud_column = min(condensed_mass_kg / area, settings.cloud_condensate_column_cap_kg_m2) * settings.cloud_suspended_fraction
    density = settings.cloud_particle_density_kg_m3 or 1000.0
    radius = settings.cloud_particle_median_radius_m
    tau = float(3.0 * cloud_column / max(2.0 * density * radius, 1e-300))
    return cloud_column, tau


def _resolved_cloud_tau(vertical: VerticalProcessResult, visible_wave: np.ndarray) -> np.ndarray | None:
    if vertical.cloud_extinction_m_inv.size == 0 or not np.any(vertical.cloud_extinction_m_inv > 0):
        return None
    tau_native = np.sum(
        vertical.cloud_extinction_m_inv * vertical.layer_thickness_m[:, None], axis=0
    )
    return np.interp(
        visible_wave,
        vertical.optical_wavelength_m,
        tau_native,
        left=float(tau_native[0]),
        right=float(tau_native[-1]),
    )


def solve_planet(*, planet: PlanetPhysicalState, star: StellarSpectrum, inventory: ElementInventory,
                 surface: SurfaceReservoirs | None = None, settings: SolverSettings | None = None,
                 database: ChemicalDatabase = BUILTIN_DATABASE) -> PlanetChemistryResult:
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
    cloud = CloudResult({}, cfg.cloud_particle_median_radius_m, 0.0, "equilibrium_bulk")
    vertical: VerticalProcessResult | None = None
    chemistry_composition: dict[str, float] = {}
    visible_wave = np.linspace(360e-9, 830e-9, 189)
    cloud_optical_wave = np.linspace(360e-9, 830e-9, 13)
    incident = _interpolate_star(star, visible_wave)
    area = 4.0 * np.pi * planet.radius_m**2

    for iteration in range(1, cfg.max_iterations + 1):
        chemistry_composition, chemistry_meta = _composition(
            inventory, cfg, temperature, planet.surface_pressure_pa, database
        )
        phase = partition_surface_reservoirs(
            planet=planet, temperature_k=temperature,
            atmospheric_mole_fractions=chemistry_composition,
            surface=reservoirs, database=database,
            activity_model=cfg.activity_model,
            liquid_phase_split=cfg.liquid_phase_split,
        )
        composition, _surface_source = atmospheric_composition_with_surface_vapor(
            planet=planet, atmospheric_mole_fractions=chemistry_composition,
            surface_vapor_mass_kg=phase.atmospheric_mass_kg, database=database,
        )
        composition_delta = (max(abs(composition.get(k, 0.0) - previous_composition.get(k, 0.0))
                                 for k in set(composition) | set(previous_composition))
                             if previous_composition is not None else 1.0)

        profile_iter = solve_isothermal_hydrostatic(
            surface_pressure_pa=planet.surface_pressure_pa,
            top_pressure_pa=cfg.top_pressure_pa,
            temperature_k=temperature,
            gravity_m_s2=planet.gravity_m_s2,
            mole_fractions=composition,
            layers=cfg.resolved_layers,
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

        condensed = sum(phase.liquid_mass_kg.values()) + sum(phase.solid_mass_kg.values())
        cloud_column, cloud_tau_bulk = _bulk_cloud_tau(
            condensed_mass_kg=condensed, planet=planet, settings=cfg
        )
        cloud_tau_spectral = (
            _resolved_cloud_tau(vertical, visible_wave)
            if cfg.cloud_mode == "lognormal_sedimentation"
            else None
        )
        if cloud_tau_spectral is not None:
            cloud_tau_for_radiation: np.ndarray | float = cloud_tau_spectral
            cloud_tau_visible = float(np.interp(550e-9, visible_wave, cloud_tau_spectral))
            cloud_model = "resolved lognormal Mie extinction + sedimentation/precipitation"
        else:
            cloud_tau_for_radiation = cloud_tau_bulk
            cloud_tau_visible = cloud_tau_bulk
            cloud_model = (
                "resolved lognormal sedimentation with bulk-gray optical fallback"
                if cfg.cloud_mode == "lognormal_sedimentation"
                else "equilibrium bulk condensate; bounded suspended fraction"
            )
        sigma_ln = np.log(cfg.cloud_particle_geometric_std)
        effective_radius = cfg.cloud_particle_median_radius_m * np.exp(2.5 * sigma_ln**2)
        cloud = CloudResult(
            {k: phase.liquid_mass_kg.get(k, 0.0) + phase.solid_mass_kg.get(k, 0.0)
             for k in set(phase.liquid_mass_kg) | set(phase.solid_mass_kg)},
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
        bond = float(np.clip(0.72 * visible_bond + 0.28 * planet.surface_albedo_initial, 0.0, 0.95))
        incoming_mean = star.bolometric_flux_w_m2 / 4.0
        absorbed = incoming_mean * (1.0 - bond)
        teff = ((absorbed + planet.internal_heat_flux_w_m2) / SIGMA_SB) ** 0.25
        tau_lw = longwave_optical_depth(
            surface_pressure_pa=planet.surface_pressure_pa,
            gravity_m_s2=planet.gravity_m_s2,
            mole_fractions=chemistry_composition,
            additional_species_column_kg_m2={
                key: 0.35 * mass / area for key, mass in phase.atmospheric_mass_kg.items()
            },
            database=database,
        )
        target = teff * (1.0 + 0.75 * tau_lw) ** 0.25
        target = float(np.clip(target, 20.0, 4000.0))
        updated = (1.0 - cfg.relaxation) * temperature + cfg.relaxation * target
        relative_t = abs(updated - temperature) / max(temperature, 1.0)
        outgoing = SIGMA_SB * teff**4
        energy_imbalance = absorbed + planet.internal_heat_flux_w_m2 - outgoing
        history.append({
            "temperature_relative": relative_t,
            "composition_absolute": composition_delta,
            "energy_w_m2": abs(energy_imbalance),
            "cloud_optical_depth_visible": float(cloud_tau_visible),
            "surface_precipitation_kg_m2_step": float(vertical.surface_precipitation_kg_m2),
        })
        temperature = updated
        previous_composition = composition
        if (relative_t <= cfg.relative_temperature_tolerance and
                composition_delta <= cfg.composition_tolerance and
                abs(energy_imbalance) <= cfg.energy_tolerance_w_m2):
            break

    if phase is None or vertical is None or previous_composition is None:
        raise RuntimeError("column solve did not produce a physical state")
    converged = (relative_t <= cfg.relative_temperature_tolerance and
                 composition_delta <= cfg.composition_tolerance and
                 abs(energy_imbalance) <= cfg.energy_tolerance_w_m2)
    profile = solve_isothermal_hydrostatic(
        surface_pressure_pa=planet.surface_pressure_pa,
        top_pressure_pa=cfg.top_pressure_pa,
        temperature_k=temperature,
        gravity_m_s2=planet.gravity_m_s2,
        mole_fractions=previous_composition,
        layers=cfg.resolved_layers,
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
        final_cloud_tau = np.full(visible_wave.shape, cloud.optical_depth_visible, dtype=float)
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
        rayleigh_reflect + (1.0 - rayleigh_reflect) * planet.surface_albedo_initial * trans,
        0.0, 0.98,
    )
    denom = np.trapezoid(incident, visible_wave)
    visible_bond = float(np.trapezoid(spectral_albedo * incident, visible_wave) / denom) if denom > 0 else planet.surface_albedo_initial
    bond = float(np.clip(0.72 * visible_bond + 0.28 * planet.surface_albedo_initial, 0.0, 0.95))
    incoming_mean = star.bolometric_flux_w_m2 / 4.0
    absorbed = incoming_mean * (1.0 - bond)
    teff = ((absorbed + planet.internal_heat_flux_w_m2) / SIGMA_SB) ** 0.25
    tau_lw = longwave_optical_depth(
        surface_pressure_pa=planet.surface_pressure_pa,
        gravity_m_s2=planet.gravity_m_s2,
        mole_fractions=chemistry_composition,
        additional_species_column_kg_m2={
            key: 0.35 * mass / area for key, mass in phase.atmospheric_mass_kg.items()
        },
        database=database,
    )
    energy_imbalance = absorbed + planet.internal_heat_flux_w_m2 - SIGMA_SB * teff**4
    thermal_wave = np.geomspace(2e-6, 80e-6, 256)
    thermal_shape = np.pi * planck_radiance_w_m2_sr_m(thermal_wave, teff)
    thermal_shape *= (SIGMA_SB * teff**4) / np.trapezoid(thermal_shape, thermal_wave)
    reflected = incident * spectral_albedo
    spectra = SpectralResult(
        visible_wave, incident, reflected, trans, tau_r, thermal_wave, thermal_shape,
        spectral_albedo, bond, min(1.5 * bond, 1.0), spectrum_to_srgb(visible_wave, reflected)
    )
    budget = EnergyBudget(
        incoming_mean, absorbed, SIGMA_SB * teff**4,
        planet.internal_heat_flux_w_m2, energy_imbalance, tau_lw
    )
    convergence = ConvergenceReport(
        converged, iteration, relative_t, composition_delta, energy_imbalance, tuple(history)
    )
    total_negative = any(
        v < 0
        for collection in (phase.atmospheric_mass_kg, phase.liquid_mass_kg, phase.solid_mass_kg)
        for v in collection.values()
    )
    diagnostics = {
        "finite": bool(
            np.isfinite(profile.pressure_pa).all()
            and np.isfinite(profile.density_kg_m3).all()
            and np.isfinite(vertical.cloud_condensate_kg_m2).all()
        ),
        "non_negative": not total_negative,
        "hydrostatic_relative_residual": profile.hydrostatic_relative_residual,
        "reservoir_mass_closure_relative": phase.mass_closure_relative,
        "vertical_process_mass_closure_relative": vertical.mass_closure_relative,
        "surface_precipitation_kg_m2_per_microphysics_step": vertical.surface_precipitation_kg_m2,
        "reevaporation_latent_cooling_w_m2_diagnostic": vertical.reevaporation_latent_cooling_w_m2,
        "chemistry": chemistry_meta,
        "fallbacks": tuple(phase.fallbacks) + tuple(vertical.fallbacks),
        "liquid_phase_count": len(phase.liquid_phases),
        "activity_model": phase.activity_model,
        "vertical_transport_mode": cfg.vertical_transport_mode,
        "cloud_process_model": vertical.model,
    }
    provenance = {
        "atmogen_version": __version__,
        "api_schema_version": API_SCHEMA_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "database_sha256": database.revision_hash,
        "fidelity": cfg.fidelity.value,
        "chemistry_mode": cfg.chemistry_mode,
        "radiation_mode": cfg.radiation_mode,
        "cloud_mode": cfg.cloud_mode,
        "activity_model_requested": cfg.activity_model,
        "liquid_phase_split": cfg.liquid_phase_split,
        "vertical_transport_mode": cfg.vertical_transport_mode,
        "eddy_diffusivity_m2_s": cfg.eddy_diffusivity_m2_s,
        "cloud_suspended_fraction": cfg.cloud_suspended_fraction,
        "cloud_condensate_column_cap_kg_m2": cfg.cloud_condensate_column_cap_kg_m2,
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
        "vertical_cloud_initialization": "uniform condensate-to-air mass ratio over a bounded suspended column",
        "latent_heat_semantics": "re-evaporation latent cooling is diagnostic redistribution until layerwise energy integration is enabled",
        "limitations": (
            "semi-gray longwave, fixed-pressure reservoir coupling, ideal-volume liquid density, "
            "bulk cold-trap depletion, reduced-order cloud source, and no layerwise radiative-convective "
            "energy integration remain approximations"
        ),
    }
    return PlanetChemistryResult(
        profile, phase, cloud, vertical, spectra, budget, convergence, diagnostics, provenance
    )


def _column_key(column: object, settings: SolverSettings, star: StellarSpectrum) -> str:
    def convert(value: object) -> object:
        if hasattr(value, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(value).items()}
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in sorted(value.items())}
        if isinstance(value, np.ndarray):
            return hashlib.sha256(value.tobytes()).hexdigest()
        return value
    raw = json.dumps({"column": convert(column), "settings": convert(settings),
                      "star_wave": hashlib.sha256(star.wavelength_m.tobytes()).hexdigest(),
                      "star_flux": hashlib.sha256(star.flux_w_m2_m.tobytes()).hexdigest()},
                     sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def solve_columns(batch: ColumnBatchInput, settings: SolverSettings | None = None) -> tuple[PlanetChemistryResult, ...]:
    """Solve a column batch with exact-state de-duplication within the batch."""
    cfg = settings or SolverSettings()
    cache: dict[str, PlanetChemistryResult] = {}
    output: list[PlanetChemistryResult] = []
    for column in batch.columns:
        key = _column_key(column, cfg, batch.star)
        if key not in cache:
            cache[key] = solve_planet(planet=column.planet, star=batch.star, inventory=column.inventory,
                                      surface=column.surface, settings=cfg)
        output.append(cache[key])
    return tuple(output)
