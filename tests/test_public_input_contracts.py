import numpy as np
import pytest

from atmogen.chemistry import IdealGibbsEquilibrium, normalized_initial_composition
from atmogen.models import PlanetPhysicalState, SolverSettings
from atmogen.radiation import (
    beer_lambert_transmission,
    blackbody_stellar_spectrum,
    longwave_optical_depth,
    planck_radiance_w_m2_sr_m,
    rayleigh_cross_section_m2,
    rayleigh_optical_depth,
)


@pytest.mark.parametrize("heat_flux", [np.nan, np.inf, -np.inf, -1.0])
def test_planet_state_rejects_invalid_internal_heat_flux(heat_flux):
    with pytest.raises(ValueError, match="internal_heat_flux"):
        PlanetPhysicalState(
            radius_m=6.4e6,
            gravity_m_s2=9.8,
            surface_pressure_pa=1.0e5,
            internal_heat_flux_w_m2=heat_flux,
        )


@pytest.mark.parametrize("top_pressure", [np.nan, np.inf, -np.inf, 0.0])
def test_solver_settings_reject_invalid_top_pressure(top_pressure):
    with pytest.raises(ValueError, match="top_pressure_pa.*finite and positive"):
        SolverSettings(top_pressure_pa=top_pressure)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_solver_settings_require_positive_integer_iterations(iterations):
    with pytest.raises((TypeError, ValueError), match="max_iterations.*integer"):
        SolverSettings(max_iterations=iterations)


@pytest.mark.parametrize(
    "name,value",
    [
        ("relative_temperature_tolerance", np.nan),
        ("relative_temperature_tolerance", np.inf),
        ("relative_temperature_tolerance", 0.0),
        ("composition_tolerance", np.nan),
        ("composition_tolerance", 0.0),
        ("energy_tolerance_w_m2", np.inf),
        ("energy_tolerance_w_m2", 0.0),
    ],
)
def test_solver_settings_reject_invalid_solver_tolerances(name, value):
    with pytest.raises(ValueError, match=f"{name}.*finite and positive"):
        SolverSettings(**{name: value})


@pytest.mark.parametrize("temperature", [np.nan, np.inf, -np.inf, 0.0])
def test_planck_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature.*finite and positive"):
        planck_radiance_w_m2_sr_m(np.asarray([500e-9]), temperature)


@pytest.mark.parametrize(
    "wave",
    [
        np.asarray([np.nan]),
        np.asarray([np.inf]),
        np.asarray([0.0]),
        np.asarray([-500e-9]),
    ],
)
def test_planck_rejects_invalid_wavelengths(wave):
    with pytest.raises(ValueError, match="wavelength.*finite and positive"):
        planck_radiance_w_m2_sr_m(wave, 5800.0)


def test_beer_lambert_rejects_nan_optical_depth():
    with pytest.raises(ValueError, match="NaN"):
        beer_lambert_transmission(np.asarray([0.1, np.nan]))


@pytest.mark.parametrize("wave", [np.asarray([0.0]), np.asarray([np.nan]), np.asarray([np.inf])])
def test_rayleigh_cross_section_rejects_invalid_wavelength(wave):
    with pytest.raises(ValueError, match="wavelength.*finite and positive"):
        rayleigh_cross_section_m2("N2", wave)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"surface_pressure_pa": np.nan, "gravity_m_s2": 9.81},
        {"surface_pressure_pa": 1.0e5, "gravity_m_s2": np.inf},
        {"surface_pressure_pa": -1.0, "gravity_m_s2": 9.81},
        {"surface_pressure_pa": 1.0e5, "gravity_m_s2": 0.0},
    ],
)
def test_rayleigh_optical_depth_rejects_invalid_column_state(kwargs):
    with pytest.raises(ValueError, match="pressure|gravity"):
        rayleigh_optical_depth(
            wavelength_m=np.asarray([450e-9, 550e-9]),
            mole_fractions={"N2": 0.8, "O2": 0.2},
            **kwargs,
        )


@pytest.mark.parametrize("composition", [{"N2": np.nan}, {"N2": -0.2, "O2": 1.2}])
def test_radiative_column_functions_reject_invalid_composition(composition):
    with pytest.raises(ValueError, match="mole fractions.*finite and non-negative"):
        rayleigh_optical_depth(
            wavelength_m=np.asarray([450e-9, 550e-9]),
            surface_pressure_pa=1.0e5,
            gravity_m_s2=9.81,
            mole_fractions=composition,
        )
    with pytest.raises(ValueError, match="mole fractions.*finite and non-negative"):
        longwave_optical_depth(
            surface_pressure_pa=1.0e5,
            gravity_m_s2=9.81,
            mole_fractions=composition,
        )


@pytest.mark.parametrize("temperature,pressure", [(np.nan, 1.0e5), (300.0, np.inf)])
def test_equilibrium_rejects_nonfinite_thermodynamic_state(temperature, pressure):
    solver = IdealGibbsEquilibrium()
    with pytest.raises(ValueError, match="temperature and pressure.*finite and positive"):
        solver.solve(
            temperature_k=temperature,
            pressure_pa=pressure,
            element_moles={"N": 2.0},
        )


@pytest.mark.parametrize("inventory", [{"N": np.nan}, {"N": -1.0}, {"": 1.0}])
def test_equilibrium_rejects_invalid_element_inventory(inventory):
    solver = IdealGibbsEquilibrium()
    with pytest.raises(ValueError, match="element inventory"):
        solver.solve(
            temperature_k=300.0,
            pressure_pa=1.0e5,
            element_moles=inventory,
        )


@pytest.mark.parametrize("hint", [{"N2": np.nan}, {"N2": -1.0}])
def test_equilibrium_rejects_invalid_initial_species_hints(hint):
    solver = IdealGibbsEquilibrium()
    with pytest.raises(ValueError, match="initial species"):
        solver.solve(
            temperature_k=300.0,
            pressure_pa=1.0e5,
            element_moles={"N": 2.0},
            initial_species_moles=hint,
        )


@pytest.mark.parametrize("state", [{"N2": np.nan}, {"N2": -1.0}, {"": 1.0}])
def test_fixed_composition_normalizer_rejects_invalid_entries(state):
    with pytest.raises(ValueError, match="initial molecular state"):
        normalized_initial_composition(state)
