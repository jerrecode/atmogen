import numpy as np
import pytest

from atmogen.hydrostatic import (
    logarithmic_pressure_interfaces,
    mean_molar_mass,
    solve_isothermal_hydrostatic,
    solve_temperature_profile_hydrostatic,
)


def test_isothermal_scale_height_and_pressure_are_analytic():
    p = solve_isothermal_hydrostatic(surface_pressure_pa=1e5, top_pressure_pa=1e2, temperature_k=300.0,
                                     gravity_m_s2=9.81, mole_fractions={"N2": 1.0}, layers=80)
    expected_top = 8.31446261815324 * 300 / (0.0280134 * 9.81) * np.log(1e5 / 1e2)
    dz = p.altitude_m[1] - p.altitude_m[0]
    assert abs((p.altitude_m[-1] + dz / 2) - expected_top) / expected_top < 2e-3
    assert np.all(np.diff(p.pressure_pa) < 0)
    assert np.all(p.density_kg_m3 > 0)
    assert p.hydrostatic_relative_residual < 4e-4



@pytest.mark.parametrize(
    "surface, top",
    [
        (np.nan, 1.0),
        (np.inf, 1.0),
        (1.0e5, np.nan),
        (1.0e5, np.inf),
    ],
)
def test_pressure_grid_rejects_nonfinite_bounds(surface, top):
    with pytest.raises(ValueError, match="finite"):
        logarithmic_pressure_interfaces(surface, top, 8)


@pytest.mark.parametrize("temperature", [np.nan, np.inf, -np.inf])
def test_isothermal_solver_rejects_nonfinite_temperature(temperature):
    with pytest.raises(ValueError, match="finite"):
        solve_isothermal_hydrostatic(
            surface_pressure_pa=1.0e5,
            top_pressure_pa=1.0e2,
            temperature_k=temperature,
            gravity_m_s2=9.81,
            mole_fractions={"N2": 1.0},
            layers=8,
        )


@pytest.mark.parametrize("gravity", [np.nan, np.inf, -np.inf])
def test_isothermal_solver_rejects_nonfinite_gravity(gravity):
    with pytest.raises(ValueError, match="finite"):
        solve_isothermal_hydrostatic(
            surface_pressure_pa=1.0e5,
            top_pressure_pa=1.0e2,
            temperature_k=300.0,
            gravity_m_s2=gravity,
            mole_fractions={"N2": 1.0},
            layers=8,
        )


@pytest.mark.parametrize(
    "composition",
    [
        {"N2": np.nan},
        {"N2": np.inf},
        {"N2": -0.1, "O2": 1.1},
    ],
)
def test_mean_molar_mass_rejects_nonfinite_or_negative_fractions(composition):
    with pytest.raises(ValueError, match="finite and non-negative"):
        mean_molar_mass(composition)


def test_variable_temperature_hydrostatic_rejects_invalid_composition():
    pi = logarithmic_pressure_interfaces(1.0e5, 1.0e2, 4)
    with pytest.raises(ValueError, match="finite and non-negative"):
        solve_temperature_profile_hydrostatic(
            pressure_interface_pa=pi,
            temperature_k=np.full(4, 280.0),
            gravity_m_s2=9.81,
            mole_fractions={"N2": 1.0, "O2": np.nan},
        )
