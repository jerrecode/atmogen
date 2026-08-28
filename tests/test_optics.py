import numpy as np

from atmogen.optics import (absorption_coefficient_m_inv, fresnel_reflectance,
                            lorentz_lorenz_mix, mie_sphere_efficiencies,
                            rayleigh_sphere_efficiencies)


def _radius_for_size_parameter(x: float, wavelength_m: float = 1.0,
                               medium_index: float = 1.0) -> float:
    return x * wavelength_m / (2.0 * np.pi * medium_index)


def test_absorption_coefficient_matches_complex_index_definition():
    wavelength = 500e-9
    index = 1.4 + 1.0e-4j
    expected = 4.0 * np.pi * 1.0e-4 / wavelength
    assert np.isclose(absorption_coefficient_m_inv(index, wavelength), expected)


def test_fresnel_normal_incidence_and_brewster_limits():
    normal = fresnel_reflectance(1.0, 1.5 + 0j, 0.0)
    expected = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
    assert np.isclose(normal.s, expected, rtol=1e-13)
    assert np.isclose(normal.p, expected, rtol=1e-13)
    assert np.isclose(normal.unpolarized, expected, rtol=1e-13)

    brewster = np.arctan(1.5)
    angled = fresnel_reflectance(1.0, 1.5 + 0j, brewster)
    assert angled.p < 1e-24
    assert angled.s > 0


def test_lorentz_lorenz_returns_identity_for_single_component_and_passive_mix():
    single = lorentz_lorenz_mix({"A": 1.33 + 2e-4j}, {"A": 1.0})
    assert np.isclose(single.real, 1.33, rtol=1e-12)
    assert np.isclose(single.imag, 2e-4, rtol=1e-10)

    mixed = lorentz_lorenz_mix(
        {"A": 1.30 + 0j, "B": 1.50 + 0.01j},
        {"A": 0.7, "B": 0.3},
    )
    assert 1.30 < mixed.real < 1.50
    assert mixed.imag > 0


def test_mie_matches_bohren_huffman_wiscombe_case_14_under_atmogen_sign_convention():
    # Literature case: x=1 and m=1.5-1j. atmogen uses n+i*kappa for a passive
    # medium, so the equivalent input is 1.5+1j.
    result = mie_sphere_efficiencies(
        radius_m=_radius_for_size_parameter(1.0),
        wavelength_m=1.0,
        particle_refractive_index=1.5 + 1.0j,
    )
    assert np.isclose(result.q_ext, 2.3363209847, rtol=2e-9)
    assert np.isclose(result.q_sca, 0.6634537615, rtol=2e-9)
    assert np.isclose(result.q_back, 0.5730025552, rtol=2e-9)
    assert np.isclose(result.asymmetry_g, 0.1921363959, rtol=2e-9)
    assert np.isclose(result.q_ext, result.q_sca + result.q_abs, rtol=1e-13)
    assert 0 < result.single_scattering_albedo < 1


def test_mie_matches_wiscombe_large_particle_regression_case():
    result = mie_sphere_efficiencies(
        radius_m=_radius_for_size_parameter(100.0),
        wavelength_m=1.0,
        particle_refractive_index=1.5 + 1.0j,
    )
    assert np.isclose(result.q_ext, 2.0975017555, rtol=2e-7)
    assert np.isclose(result.q_sca, 1.2836970494, rtol=2e-7)
    assert np.isclose(result.asymmetry_g, 0.8502519977, rtol=2e-7)
    assert result.series_terms > 100


def test_full_mie_converges_to_rayleigh_small_particle_limit():
    wavelength = 550e-9
    x = 2.0e-3
    radius = _radius_for_size_parameter(x, wavelength)
    rayleigh = rayleigh_sphere_efficiencies(
        radius_m=radius,
        wavelength_m=wavelength,
        particle_refractive_index=1.5 + 0.01j,
    )
    # Force the full series above its internal Rayleigh shortcut by using x=2e-3.
    full = mie_sphere_efficiencies(
        radius_m=radius,
        wavelength_m=wavelength,
        particle_refractive_index=1.5 + 0.01j,
    )
    assert np.isclose(full.q_sca, rayleigh.q_sca, rtol=2e-4)
    assert np.isclose(full.q_abs, rayleigh.q_abs, rtol=2e-3)
    assert abs(full.asymmetry_g) < 1e-5


def test_nonabsorbing_particle_conserves_extinction_as_scattering():
    result = mie_sphere_efficiencies(
        radius_m=0.4e-6,
        wavelength_m=550e-9,
        particle_refractive_index=1.33 + 0j,
    )
    assert result.q_abs < 1e-11
    assert np.isclose(result.q_ext, result.q_sca, rtol=1e-11, atol=1e-13)
    assert np.isclose(result.single_scattering_albedo, 1.0, atol=1e-12)
