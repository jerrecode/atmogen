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
from atmogen.hydrostatic import (
    logarithmic_cell_mean_pressure,
    logarithmic_pressure_interfaces,
    solve_temperature_profile_hydrostatic,
)
from atmogen.thermal import (
    dry_adiabatic_log_pressure_gradient,
    dry_convective_adjustment,
    gray_radiative_temperature_profile,
)


def test_gray_profile_is_surface_anchored_and_cools_upward():
    interfaces = logarithmic_pressure_interfaces(1.0e5, 10.0, 24)
    pressure = logarithmic_cell_mean_pressure(interfaces)
    temperature = gray_radiative_temperature_profile(
        pressure_pa=pressure,
        surface_pressure_pa=1.0e5,
        surface_temperature_k=300.0,
        longwave_optical_depth_surface=3.0,
        optical_depth_pressure_exponent=2.0,
    )
    assert temperature[0] == 300.0
    assert temperature[-1] < temperature[0]
    assert np.all(np.diff(temperature) <= 0.0)
    assert np.all(np.isfinite(temperature))
    assert np.all(temperature > 0.0)


def test_dry_convective_adjustment_caps_superadiabatic_gradient():
    pressure = np.geomspace(1.0e5, 1.0e3, 30)
    # Deliberately steeper than the N2 dry adiabat.
    unstable = 300.0 * (pressure / pressure[0]) ** 0.50
    adjusted, changed = dry_convective_adjustment(
        pressure_pa=pressure,
        temperature_k=unstable,
        mole_fractions={"N2": 1.0},
    )
    nabla_ad = dry_adiabatic_log_pressure_gradient({"N2": 1.0})
    gradients = np.diff(np.log(adjusted)) / np.diff(np.log(pressure))
    assert np.any(changed)
    assert not changed[0]
    assert np.max(gradients) <= nabla_ad + 2e-12
    assert np.all(adjusted >= unstable)


def test_nonisothermal_hydrostatic_profile_closes_column_mass():
    interfaces = logarithmic_pressure_interfaces(1.0e5, 100.0, 40)
    pressure = logarithmic_cell_mean_pressure(interfaces)
    temperature = 290.0 - 70.0 * np.linspace(0.0, 1.0, pressure.size)
    profile = solve_temperature_profile_hydrostatic(
        pressure_interface_pa=interfaces,
        temperature_k=temperature,
        gravity_m_s2=9.81,
        mole_fractions={"N2": 0.8, "O2": 0.2},
    )
    assert profile.hydrostatic_relative_residual < 2e-15
    assert np.all(np.diff(profile.altitude_m) > 0.0)
    assert np.all(profile.density_kg_m3 > 0.0)
    assert np.array_equal(profile.temperature_k, temperature)


def _earth_case(fidelity: Fidelity):
    species = {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}
    return solve_planet(
        planet=PlanetPhysicalState(6_371_000.0, 9.80665, 101325.0),
        star=blackbody_stellar_spectrum(5772.0, 1361.0),
        inventory=ElementInventory(
            species_moles_to_elements(species),
            species,
            "thermal-profile regression",
        ),
        surface=SurfaceReservoirs({"H2O": 1.4e21}),
        settings=SolverSettings(
            fidelity=fidelity,
            vertical_layers=20,
            chemistry_mode="fixed_species",
            max_iterations=40,
        ),
    )


def test_fast_solver_retains_isothermal_profile_contract():
    result = _earth_case(Fidelity.FAST)
    assert result.convergence.converged
    assert result.diagnostics["temperature_profile_model"] == "isothermal_fast"
    assert result.diagnostics["dry_convective_adjusted_layers"] == 0
    assert np.ptp(result.atmosphere.temperature_k) == 0.0


def test_standard_solver_uses_stable_nonisothermal_dry_profile():
    result = _earth_case(Fidelity.STANDARD)
    assert result.convergence.converged
    assert result.diagnostics["temperature_profile_model"] == "dry_gray_radiative_convective"
    assert result.diagnostics["temperature_profile_range_k"] > 1.0
    assert result.atmosphere.temperature_k[0] > result.atmosphere.temperature_k[-1]
    assert result.atmosphere.hydrostatic_relative_residual < 2e-12

    nabla_ad = dry_adiabatic_log_pressure_gradient(
        result.atmosphere.mole_fractions
    )
    gradients = np.diff(np.log(result.atmosphere.temperature_k)) / np.diff(
        np.log(result.atmosphere.pressure_pa)
    )
    assert np.max(gradients) <= nabla_ad + 2e-12
