from __future__ import annotations

"""Layer-resolved transport/cloud diagnostics used by the coupled column solver.

This module deliberately keeps the current reduced-order cloud source separate from
microphysical transport.  The suspended cloud column is a bounded fraction of the
thermodynamically condensed reservoir; sedimentation and re-evaporation redistribute
that suspended column and therefore do not create a second material inventory.
"""

from typing import Mapping

import numpy as np

from .cloud_microphysics import (
    ParticlePopulation,
    particle_optical_coefficients,
    precipitation_step,
    sedimentation_mass_flux,
)
from .database import BUILTIN_DATABASE, ChemicalDatabase
from .models import (
    AtmosphericProfile,
    PhaseReservoirResult,
    PlanetPhysicalState,
    SolverSettings,
    VerticalProcessResult,
)


def layer_thickness_from_centres(altitude_m: np.ndarray) -> np.ndarray:
    """Construct positive finite-volume layer thicknesses from cell-centre altitude."""
    z = np.asarray(altitude_m, dtype=float)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("altitude_m must be a finite strictly increasing 1-D array")
    edges = np.empty(z.size + 1, dtype=float)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[-1] = z[-1] + max(z[-1] - edges[-2], 0.5 * (z[-1] - z[-2]))
    thickness = np.diff(edges)
    if np.any(thickness <= 0):
        raise RuntimeError("constructed non-positive vertical layer thickness")
    return thickness


def _condensate_density_kg_m3(
    phase: PhaseReservoirResult,
    database: ChemicalDatabase,
    fallback_density_kg_m3: float | None,
) -> tuple[float, tuple[str, ...]]:
    masses = {
        key: float(phase.liquid_mass_kg.get(key, 0.0) + phase.solid_mass_kg.get(key, 0.0))
        for key in set(phase.liquid_mass_kg) | set(phase.solid_mass_kg)
    }
    masses = {key: value for key, value in masses.items() if value > 0}
    known_volume = 0.0
    known_mass = 0.0
    missing: list[str] = []
    for key, mass in masses.items():
        density = database.get(key).liquid_density_kg_m3
        if density is None or density <= 0:
            missing.append(key)
            continue
        known_mass += mass
        known_volume += mass / density
    if known_mass > 0 and known_volume > 0 and not missing:
        return float(known_mass / known_volume), ()
    if fallback_density_kg_m3 is not None:
        value = float(fallback_density_kg_m3)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("cloud_particle_density_kg_m3 must be finite and positive")
        note = (
            "cloud particle density uses configured fallback because condensed material "
            f"density is incomplete: {', '.join(sorted(missing)) or 'mixed phase'}"
        )
        return value, (note,)
    if known_mass > 0 and known_volume > 0:
        return float(known_mass / known_volume), (
            "cloud particle density ignores components with unavailable condensed density: "
            + ", ".join(sorted(missing)),
        )
    return 1000.0, (
        "cloud particle density uses generic bounded estimate 1000 kg/m3 because no "
        "condensed-density data are available",
    )


def solve_vertical_processes(
    *,
    profile: AtmosphericProfile,
    phase: PhaseReservoirResult,
    planet: PlanetPhysicalState,
    settings: SolverSettings,
    optical_wavelength_m: np.ndarray,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> VerticalProcessResult:
    """Resolve Kzz timescales, suspended cloud settling, Mie optics and precipitation.

    Condensate is initially distributed with constant condensate-to-air mass ratio.
    This is a documented reduced-order initialization, not a nucleation/cloud-base
    calculation.  The microphysical transport itself is conservative.
    """
    thickness = layer_thickness_from_centres(profile.altitude_m)
    layers = thickness.size
    if settings.vertical_transport_mode == "eddy_diffusion":
        kzz = np.full(layers, float(settings.eddy_diffusivity_m2_s), dtype=float)
        mixing_time = thickness**2 / np.maximum(kzz, 1e-300)
    else:
        kzz = np.zeros(layers, dtype=float)
        mixing_time = np.full(layers, np.inf, dtype=float)

    optical_wave = np.asarray(optical_wavelength_m, dtype=float)
    if optical_wave.ndim != 1 or optical_wave.size < 1 or np.any(~np.isfinite(optical_wave)) or np.any(optical_wave <= 0):
        raise ValueError("optical_wavelength_m must be a finite positive 1-D array")
    ext = np.zeros((layers, optical_wave.size), dtype=float)
    sca = np.zeros_like(ext)
    abs_ = np.zeros_like(ext)
    omega = np.zeros_like(ext)
    asymmetry = np.zeros_like(ext)
    condensate_column = np.zeros(layers, dtype=float)
    mass_concentration = np.zeros(layers, dtype=float)
    number_concentration = np.zeros(layers, dtype=float)
    settling_velocity = np.zeros(layers, dtype=float)
    sedimentation_flux = np.zeros(layers, dtype=float)
    fallbacks: list[str] = []

    total_condensed_kg = float(sum(phase.liquid_mass_kg.values()) + sum(phase.solid_mass_kg.values()))
    area_m2 = 4.0 * np.pi * planet.radius_m**2
    available_column = total_condensed_kg / area_m2
    suspended_total = min(available_column, float(settings.cloud_condensate_column_cap_kg_m2)) * float(settings.cloud_suspended_fraction)

    precipitation_downward = np.zeros(layers, dtype=float)
    precipitation_reevap = np.zeros(layers, dtype=float)
    surface_precip = 0.0
    closure = 0.0
    latent_reevap_w_m2 = 0.0
    model = "no suspended condensate"

    if suspended_total > 0 and settings.cloud_mode == "lognormal_sedimentation":
        air_column_weights = np.asarray(profile.density_kg_m3, dtype=float) * thickness
        weight_sum = float(np.sum(air_column_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise RuntimeError("atmospheric column mass weights are invalid")
        condensate_column = suspended_total * air_column_weights / weight_sum
        mass_concentration = condensate_column / thickness

        density, density_notes = _condensate_density_kg_m3(
            phase, database, settings.cloud_particle_density_kg_m3
        )
        fallbacks.extend(density_notes)
        sigma_g = float(settings.cloud_particle_geometric_std)
        radius = float(settings.cloud_particle_median_radius_m)
        unit_population = ParticlePopulation(
            composition={"condensate": 1.0},
            number_concentration_m3=1.0,
            median_radius_m=radius,
            geometric_std=sigma_g,
            particle_density_kg_m3=density,
        )
        mean_particle_mass = unit_population.mass_concentration_kg_m3
        number_concentration = mass_concentration / max(mean_particle_mass, 1e-300)

        have_mie_index = settings.cloud_refractive_index_real is not None
        if not have_mie_index:
            fallbacks.append(
                "resolved cloud Mie optics disabled because no complex refractive index was supplied; "
                "radiation uses the explicit bulk-gray cloud fallback"
            )
        refractive_index = None
        if have_mie_index:
            refractive_index = complex(
                float(settings.cloud_refractive_index_real),
                float(settings.cloud_refractive_index_imag),
            )

        for layer in range(layers):
            if mass_concentration[layer] <= 0:
                continue
            population = ParticlePopulation(
                composition={"condensate": 1.0},
                number_concentration_m3=float(number_concentration[layer]),
                median_radius_m=radius,
                geometric_std=sigma_g,
                particle_density_kg_m3=density,
            )
            settled = sedimentation_mass_flux(
                population=population,
                gas_density_kg_m3=max(float(profile.density_kg_m3[layer]), 1e-12),
                gas_dynamic_viscosity_pa_s=float(settings.gas_dynamic_viscosity_pa_s),
                gravity_m_s2=planet.gravity_m_s2,
                layer_depth_m=float(thickness[layer]),
                quadrature_order=int(settings.cloud_quadrature_order),
            )
            settling_velocity[layer] = settled.mass_weighted_velocity_m_s
            sedimentation_flux[layer] = settled.mass_flux_kg_m2_s
            if refractive_index is not None:
                optics = particle_optical_coefficients(
                    population=population,
                    wavelength_m=optical_wave,
                    particle_refractive_index=refractive_index,
                    quadrature_order=int(settings.cloud_quadrature_order),
                )
                ext[layer] = optics.extinction_m_inv
                sca[layer] = optics.scattering_m_inv
                abs_[layer] = optics.absorption_m_inv
                omega[layer] = optics.single_scattering_albedo
                asymmetry[layer] = optics.asymmetry_g

        evap = None
        if settings.cloud_reevaporation_timescale_s is not None:
            evap = np.full(layers, float(settings.cloud_reevaporation_timescale_s), dtype=float)
        precip = precipitation_step(
            condensate_kg_m2=condensate_column,
            layer_thickness_m=thickness,
            settling_velocity_m_s=settling_velocity,
            timestep_s=float(settings.cloud_microphysics_timestep_s),
            evaporation_timescale_s=evap,
        )
        condensate_column = precip.remaining_condensate_kg_m2
        precipitation_downward = precip.downward_transfer_kg_m2
        precipitation_reevap = precip.reevaporated_kg_m2
        surface_precip = precip.surface_precipitation_kg_m2
        closure = precip.mass_closure_relative
        latent_by_species = []
        total_phase_mass = max(total_condensed_kg, 1e-300)
        for key in set(phase.liquid_mass_kg) | set(phase.solid_mass_kg):
            species_mass = float(phase.liquid_mass_kg.get(key, 0.0) + phase.solid_mass_kg.get(key, 0.0))
            latent = database.get(key).latent_heat_j_kg
            if species_mass > 0 and latent is not None and latent > 0:
                latent_by_species.append((species_mass / total_phase_mass) * latent)
        if latent_by_species and settings.cloud_microphysics_timestep_s > 0:
            latent_mix = float(sum(latent_by_species))
            latent_reevap_w_m2 = float(np.sum(precipitation_reevap) * latent_mix / settings.cloud_microphysics_timestep_s)
        model = "bounded suspended condensate + lognormal sedimentation/precipitation"
    elif suspended_total > 0:
        model = "equilibrium bulk condensate"

    return VerticalProcessResult(
        layer_thickness_m=thickness,
        eddy_diffusivity_m2_s=kzz,
        mixing_timescale_s=mixing_time,
        cloud_condensate_kg_m2=condensate_column,
        cloud_mass_concentration_kg_m3=mass_concentration,
        cloud_number_concentration_m3=number_concentration,
        cloud_settling_velocity_m_s=settling_velocity,
        cloud_sedimentation_flux_kg_m2_s=sedimentation_flux,
        optical_wavelength_m=optical_wave.copy(),
        cloud_extinction_m_inv=ext,
        cloud_scattering_m_inv=sca,
        cloud_absorption_m_inv=abs_,
        cloud_single_scattering_albedo=omega,
        cloud_asymmetry_g=asymmetry,
        precipitation_downward_kg_m2=precipitation_downward,
        precipitation_reevaporated_kg_m2=precipitation_reevap,
        surface_precipitation_kg_m2=float(surface_precip),
        reevaporation_latent_cooling_w_m2=float(latent_reevap_w_m2),
        mass_closure_relative=float(closure),
        model=model,
        fallbacks=tuple(fallbacks),
    )
