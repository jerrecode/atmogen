import numpy as np

from atmogen.cloud_microphysics import (ParticlePopulation,
                                        cunningham_slip_correction,
                                        particle_optical_coefficients,
                                        precipitation_step,
                                        sedimentation_mass_flux,
                                        sphere_drag_coefficient,
                                        terminal_settling_velocity)
from atmogen.optics import mie_sphere_efficiencies


def test_lognormal_particle_moments_and_mass_concentration_are_analytic():
    population = ParticlePopulation(
        composition={"H2O": 1.0},
        number_concentration_m3=1.0e8,
        median_radius_m=2.0e-6,
        geometric_std=1.5,
        particle_density_kg_m3=1000.0,
    )
    sigma_ln = np.log(1.5)
    expected_r3 = (2.0e-6) ** 3 * np.exp(0.5 * 9.0 * sigma_ln**2)
    assert np.isclose(population.radius_moment(3), expected_r3, rtol=1e-14)
    expected_mass = 1.0e8 * 1000.0 * 4.0 * np.pi / 3.0 * expected_r3
    assert np.isclose(population.mass_concentration_kg_m3, expected_mass, rtol=1e-14)
    assert population.effective_radius_m > population.median_radius_m


def test_stokes_cunningham_branch_matches_creeping_flow_formula():
    radius = 1.0e-6
    rho_p = 1000.0
    rho_g = 1.2
    viscosity = 1.8e-5
    gravity = 9.81
    mean_free_path = 65e-9
    correction = cunningham_slip_correction(radius, mean_free_path)
    expected = 2.0 / 9.0 * (rho_p - rho_g) * gravity * radius**2 / viscosity * correction
    result = terminal_settling_velocity(
        particle_radius_m=radius,
        particle_density_kg_m3=rho_p,
        gas_density_kg_m3=rho_g,
        gas_dynamic_viscosity_pa_s=viscosity,
        gravity_m_s2=gravity,
        mean_free_path_m=mean_free_path,
    )
    assert result.regime == "Stokes-Cunningham"
    assert result.reynolds_number < 0.1
    assert np.isclose(result.terminal_velocity_m_s, expected, rtol=1e-14)
    assert result.cunningham_factor > 1.0


def test_larger_particle_switches_to_nonlinear_drag_regime():
    result = terminal_settling_velocity(
        particle_radius_m=100e-6,
        particle_density_kg_m3=1000.0,
        gas_density_kg_m3=1.2,
        gas_dynamic_viscosity_pa_s=1.8e-5,
        gravity_m_s2=9.81,
    )
    assert result.regime == "Schiller-Naumann"
    assert result.reynolds_number > 0.1
    assert result.terminal_velocity_m_s > 0
    assert np.isclose(result.drag_coefficient, sphere_drag_coefficient(result.reynolds_number))


def test_monodisperse_sedimentation_flux_is_mass_concentration_times_velocity():
    population = ParticlePopulation(
        composition={"H2O": 1.0},
        number_concentration_m3=2.0e7,
        median_radius_m=5e-6,
        geometric_std=1.0,
        particle_density_kg_m3=1000.0,
    )
    settling = terminal_settling_velocity(
        particle_radius_m=population.median_radius_m,
        particle_density_kg_m3=population.particle_density_kg_m3,
        gas_density_kg_m3=1.0,
        gas_dynamic_viscosity_pa_s=1.8e-5,
        gravity_m_s2=9.81,
    )
    result = sedimentation_mass_flux(
        population=population,
        gas_density_kg_m3=1.0,
        gas_dynamic_viscosity_pa_s=1.8e-5,
        gravity_m_s2=9.81,
        layer_depth_m=1000.0,
        quadrature_order=1,
    )
    assert np.isclose(result.mass_concentration_kg_m3, population.mass_concentration_kg_m3, rtol=1e-14)
    assert np.isclose(result.mass_weighted_velocity_m_s, settling.terminal_velocity_m_s, rtol=1e-14)
    assert np.isclose(result.mass_flux_kg_m2_s,
                      result.mass_concentration_kg_m3 * settling.terminal_velocity_m_s,
                      rtol=1e-14)
    assert np.isclose(result.residence_time_s, 1000.0 / settling.terminal_velocity_m_s)


def test_monodisperse_mie_population_matches_single_particle_cross_section():
    radius = 0.2e-6
    number = 3.0e8
    wavelength = np.asarray([450e-9, 550e-9, 700e-9])
    index = 1.33 + 0.0j
    population = ParticlePopulation(
        composition={"H2O": 1.0},
        number_concentration_m3=number,
        median_radius_m=radius,
        geometric_std=1.0,
        particle_density_kg_m3=1000.0,
    )
    optical = particle_optical_coefficients(
        population=population,
        wavelength_m=wavelength,
        particle_refractive_index=index,
        quadrature_order=1,
    )
    for i, wave in enumerate(wavelength):
        mie = mie_sphere_efficiencies(
            radius_m=radius,
            wavelength_m=float(wave),
            particle_refractive_index=index,
        )
        expected_ext = number * np.pi * radius**2 * mie.q_ext
        assert np.isclose(optical.extinction_m_inv[i], expected_ext, rtol=2e-12)
        assert np.isclose(optical.scattering_m_inv[i], optical.extinction_m_inv[i], rtol=2e-11)
        assert optical.absorption_m_inv[i] < 1e-16
        assert np.isclose(optical.single_scattering_albedo[i], 1.0, atol=2e-11)


def test_precipitation_step_conserves_mass_and_routes_only_one_layer_per_step():
    initial = np.asarray([1.0, 2.0, 4.0])
    result = precipitation_step(
        condensate_kg_m2=initial,
        layer_thickness_m=np.asarray([100.0, 100.0, 100.0]),
        settling_velocity_m_s=np.asarray([100.0, 100.0, 100.0]),
        timestep_s=100.0,
    )
    # exp(-100) is negligible: bottom mass reaches surface, each higher layer drops
    # by exactly one represented layer, not through the whole column in one call.
    assert np.isclose(result.surface_precipitation_kg_m2, 1.0, rtol=0, atol=1e-12)
    assert np.allclose(result.remaining_condensate_kg_m2, [2.0, 4.0, 0.0], atol=1e-12)
    assert np.allclose(result.reevaporated_kg_m2, 0.0)
    assert result.mass_closure_relative < 1e-14


def test_precipitation_re_evaporates_in_undersaturated_receiving_layer():
    result = precipitation_step(
        condensate_kg_m2=np.asarray([0.0, 1.0]),
        layer_thickness_m=np.asarray([100.0, 100.0]),
        settling_velocity_m_s=np.asarray([0.0, 10.0]),
        timestep_s=100.0,
        evaporation_timescale_s=np.asarray([1.0, np.inf]),
    )
    assert result.surface_precipitation_kg_m2 == 0.0
    assert result.reevaporated_kg_m2[0] > 0.999
    assert result.remaining_condensate_kg_m2[0] < 1e-3
    assert result.mass_closure_relative < 1e-14
