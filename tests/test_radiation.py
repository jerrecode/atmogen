import numpy as np

from atmogen.radiation import (beer_lambert_transmission, fresnel_normal_reflectance,
                               planck_radiance_w_m2_sr_m, rayleigh_cross_section_m2)


def test_beer_lambert_and_zero_opacity_limits():
    assert np.allclose(beer_lambert_transmission([0.0, 1.0]), [1.0, np.e**-1])


def test_rayleigh_lambda_minus_four_trend():
    sigma = rayleigh_cross_section_m2("N2", np.asarray([400e-9, 800e-9]))
    assert np.isclose(sigma[0] / sigma[1], 16.0, rtol=1e-12)


def test_fresnel_normal_incidence_limits():
    assert fresnel_normal_reflectance(1 + 0j, 1 + 0j) == 0
    assert np.isclose(fresnel_normal_reflectance(1 + 0j, 1.5 + 0j), 0.04)


def test_planck_peak_moves_to_shorter_wavelength_when_hotter():
    wave = np.geomspace(1e-6, 50e-6, 5000)
    cold = wave[np.argmax(planck_radiance_w_m2_sr_m(wave, 250.0))]
    warm = wave[np.argmax(planck_radiance_w_m2_sr_m(wave, 500.0))]
    assert warm < cold
