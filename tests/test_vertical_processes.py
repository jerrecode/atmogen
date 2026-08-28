import numpy as np

from atmogen import (
    ElementInventory,
    Fidelity,
    PlanetPhysicalState,
    SolverSettings,
    SurfaceReservoirs,
    blackbody_stellar_spectrum,
    solve_planet,
    species_moles_to_elements,
)


def _earthlike(settings: SolverSettings):
    species = {"N2": 0.78, "O2": 0.21, "Ar": 0.009, "CO2": 0.001}
    return solve_planet(
        planet=PlanetPhysicalState(6.371e6, 9.80665, 101325.0, 288.15, 0.15),
        star=blackbody_stellar_spectrum(5772.0, 1361.0),
        inventory=ElementInventory(
            species_moles_to_elements(species), species,
            semantics="vertical-process-regression",
        ),
        surface=SurfaceReservoirs({"H2O": 1.0e18}),
        settings=settings,
    )


def test_resolved_vertical_processes_are_finite_and_mass_conserving():
    result = _earthlike(SolverSettings(
        fidelity=Fidelity.FAST,
        vertical_layers=12,
        chemistry_mode="fixed_species",
        cloud_mode="lognormal_sedimentation",
        vertical_transport_mode="eddy_diffusion",
        eddy_diffusivity_m2_s=50.0,
        cloud_particle_median_radius_m=8e-6,
        cloud_particle_geometric_std=1.35,
        cloud_refractive_index_real=1.333,
        cloud_refractive_index_imag=1e-8,
        cloud_microphysics_timestep_s=600.0,
        cloud_quadrature_order=8,
    ))
    vertical = result.vertical
    assert vertical.layer_thickness_m.shape == (12,)
    assert np.all(vertical.layer_thickness_m > 0)
    assert np.all(vertical.eddy_diffusivity_m2_s == 50.0)
    assert np.all(np.isfinite(vertical.mixing_timescale_s))
    assert vertical.cloud_extinction_m_inv.shape[0] == 12
    assert np.all(vertical.cloud_extinction_m_inv >= 0)
    assert np.all((vertical.cloud_single_scattering_albedo >= 0) &
                  (vertical.cloud_single_scattering_albedo <= 1))
    assert vertical.mass_closure_relative < 2e-12
    assert vertical.surface_precipitation_kg_m2 >= 0
    assert result.clouds.model.startswith("resolved lognormal")
    assert result.clouds.optical_depth_visible >= 0


def test_resolved_cloud_without_index_records_gray_fallback():
    result = _earthlike(SolverSettings(
        vertical_layers=10,
        chemistry_mode="fixed_species",
        cloud_mode="lognormal_sedimentation",
        cloud_refractive_index_real=None,
        cloud_microphysics_timestep_s=60.0,
        cloud_quadrature_order=4,
    ))
    assert any("Mie optics disabled" in item for item in result.vertical.fallbacks)
    assert result.clouds.optical_depth_visible >= 0


def test_bulk_mode_preserves_zero_resolved_precipitation():
    result = _earthlike(SolverSettings(
        vertical_layers=8,
        chemistry_mode="fixed_species",
        cloud_mode="equilibrium_bulk",
    ))
    assert result.vertical.surface_precipitation_kg_m2 == 0.0
    assert np.all(result.vertical.cloud_sedimentation_flux_kg_m2_s == 0)
