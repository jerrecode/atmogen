import numpy as np

from atmogen.database import ProvenanceClass, Reaction
from atmogen.kinetics import integrate_kinetics
from atmogen.photochemistry import (H_PLANCK, C_LIGHT, PhotolysisData,
                                     attenuate_actinic_photon_flux,
                                     column_photolysis_rates_s1,
                                     photolysis_rate_s1,
                                     spectral_irradiance_to_photon_flux)


def synthetic_photolysis_data() -> PhotolysisData:
    wavelength = np.linspace(200e-9, 300e-9, 101)
    return PhotolysisData(
        reaction_key="co2_photolysis_test",
        parent_species="CO2",
        wavelength_m=wavelength,
        cross_section_m2=np.full_like(wavelength, 1.0e-21),
        quantum_yield=np.ones_like(wavelength),
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic constant-spectrum analytical regression data",
        validity="test only",
        temperature_k=300.0,
    )


def test_spectral_irradiance_to_photon_flux_matches_photon_energy_definition():
    wavelength = np.asarray([300e-9, 600e-9])
    irradiance = np.asarray([2.0, 5.0])
    flux = spectral_irradiance_to_photon_flux(wavelength, irradiance)
    expected = irradiance * wavelength / (H_PLANCK * C_LIGHT)
    assert np.allclose(flux, expected, rtol=1e-14)


def test_constant_cross_section_photolysis_has_analytic_integral():
    data = synthetic_photolysis_data()
    photon_flux = np.full_like(data.wavelength_m, 1.0e25)
    result = photolysis_rate_s1(data, photon_flux)
    expected = 1.0e-21 * 1.0e25 * (300e-9 - 200e-9)
    assert np.isclose(result, expected, rtol=1e-13)


def test_actinic_attenuation_and_column_photolysis_are_monotonic_with_depth():
    data = synthetic_photolysis_data()
    top_flux = np.full_like(data.wavelength_m, 1.0e25)
    tau = np.vstack((
        np.zeros_like(data.wavelength_m),
        np.full_like(data.wavelength_m, 0.5),
        np.full_like(data.wavelength_m, 2.0),
    ))
    local_flux = attenuate_actinic_photon_flux(top_flux, tau)
    assert np.all(local_flux[0] > local_flux[1])
    assert np.all(local_flux[1] > local_flux[2])
    rates = column_photolysis_rates_s1(data, top_flux, tau)
    assert rates.shape == (3,)
    assert rates[0] > rates[1] > rates[2] > 0
    assert np.isclose(rates[1] / rates[0], np.exp(-0.5), rtol=1e-12)


def test_spectral_photolysis_rate_drives_analytic_first_order_kinetics():
    data = synthetic_photolysis_data()
    top_flux = np.full_like(data.wavelength_m, 1.0e25)
    j_rate = photolysis_rate_s1(data, top_flux)
    reaction = Reaction(
        key=data.reaction_key,
        reactants={"CO2": 1.0},
        products={"CO": 1.0, "O2": 0.5},
        rate_law="photolysis",
        rate_coefficient_units="s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic analytical photolysis regression reaction",
        validity="test only",
    )
    duration = 1000.0
    result = integrate_kinetics(
        reactions=(reaction,),
        initial_concentration_mol_m3={"CO2": 1.0},
        temperature_k=300.0,
        duration_s=duration,
        photolysis_rates_s1={reaction.key: j_rate},
        relative_tolerance=1e-10,
        absolute_tolerance_mol_m3=1e-13,
    )
    expected = np.exp(-j_rate * duration)
    assert np.isclose(result.final_concentration_mol_m3["CO2"], expected, rtol=2e-8)
    assert np.isclose(result.final_concentration_mol_m3["CO"], 1.0 - expected, rtol=2e-8)
    assert np.isclose(result.final_concentration_mol_m3["O2"], 0.5 * (1.0 - expected), rtol=2e-8)
    assert result.element_relative_residual < 2e-9


def test_photolysis_interpolates_external_spectral_grid_without_extrapolating():
    data = synthetic_photolysis_data()
    external_wave = np.linspace(150e-9, 350e-9, 401)
    external_flux = np.full_like(external_wave, 2.0e25)
    rate = photolysis_rate_s1(data, external_flux, wavelength_m=external_wave)
    expected = 1.0e-21 * 2.0e25 * (300e-9 - 200e-9)
    assert np.isclose(rate, expected, rtol=2e-13)
