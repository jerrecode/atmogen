import numpy as np
import pytest

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
    dilute_saturated_convective_adjustment,
    dilute_saturated_log_pressure_gradient,
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


def test_dilute_saturated_gradient_is_shallower_than_dry_for_warm_water():
    gradient, r_s, note = dilute_saturated_log_pressure_gradient(
        pressure_pa=1.0e5,
        temperature_k=300.0,
        mole_fractions={"N2": 0.98, "H2O": 0.02},
        condensible="H2O",
    )
    dry = dry_adiabatic_log_pressure_gradient({"N2": 1.0})
    assert note is None
    assert gradient is not None and r_s is not None
    assert 0 < gradient < dry
    assert 0 < r_s < 0.25


def test_dilute_saturated_backend_rejects_non_dilute_water_state():
    gradient, r_s, note = dilute_saturated_log_pressure_gradient(
        pressure_pa=1.0e5,
        temperature_k=373.15,
        mole_fractions={"N2": 0.5, "H2O": 0.5},
        condensible="H2O",
    )
    assert gradient is None
    assert r_s is None
    assert note and "reaches/exceeds total pressure" in note


def test_estimated_methane_saturation_requires_explicit_permission():
    gradient, _r_s, note = dilute_saturated_log_pressure_gradient(
        pressure_pa=1.5e5,
        temperature_k=100.0,
        mole_fractions={"N2": 0.95, "CH4": 0.05},
        condensible="CH4",
    )
    assert gradient is None
    assert note and "estimated saturation-pressure" in note

    allowed, r_s, allowed_note = dilute_saturated_log_pressure_gradient(
        pressure_pa=1.5e5,
        temperature_k=100.0,
        mole_fractions={"N2": 0.95, "CH4": 0.05},
        condensible="CH4",
        allow_estimated_saturation=True,
    )
    assert allowed is not None and allowed > 0
    assert r_s is not None and 0 <= r_s <= 0.25
    assert allowed_note and "estimated" in allowed_note


def test_saturated_adjustment_uses_moist_constraint_then_dry_above_freezing_transition():
    pressure = np.geomspace(1.0e5, 1.0e4, 30)
    unstable = 300.0 * (pressure / pressure[0]) ** 0.45
    adjusted, changed, moist_used, notes = dilute_saturated_convective_adjustment(
        pressure_pa=pressure,
        temperature_k=unstable,
        mole_fractions={"N2": 0.98, "H2O": 0.02},
        condensible="H2O",
    )
    assert np.any(changed)
    assert np.any(moist_used)
    assert np.all(adjusted >= unstable)
    assert any("solid regime" in note for note in notes)


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


def _earth_case(fidelity: Fidelity, **settings_overrides):
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
            **settings_overrides,
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


def test_high_solver_automatically_uses_water_saturated_constraint():
    result = _earth_case(Fidelity.HIGH)
    assert result.convergence.converged
    assert result.diagnostics["temperature_profile_model"] == (
        "dilute_saturated_gray_radiative_convective"
    )
    assert result.diagnostics["moist_condensible"] == "H2O"
    assert result.diagnostics["saturated_convective_constraint_layers"] > 0
    assert result.atmosphere.temperature_k[0] > result.atmosphere.temperature_k[-1]
    assert result.atmosphere.hydrostatic_relative_residual < 2e-12


def test_explicit_saturated_mode_can_be_strict_when_no_condensate_exists():
    species = {"N2": 1.0}
    with pytest.raises(RuntimeError, match="not physically eligible"):
        solve_planet(
            planet=PlanetPhysicalState(6_371_000.0, 9.80665, 101325.0),
            star=blackbody_stellar_spectrum(5772.0, 1361.0),
            inventory=ElementInventory(
                species_moles_to_elements(species), species, "dry strict regression"
            ),
            settings=SolverSettings(
                fidelity=Fidelity.HIGH,
                vertical_layers=16,
                chemistry_mode="fixed_species",
                temperature_profile_mode="dilute_saturated",
                allow_fidelity_fallback=False,
            ),
        )
