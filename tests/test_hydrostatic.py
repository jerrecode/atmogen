import numpy as np

from atmogen.hydrostatic import solve_isothermal_hydrostatic


def test_isothermal_scale_height_and_pressure_are_analytic():
    p = solve_isothermal_hydrostatic(surface_pressure_pa=1e5, top_pressure_pa=1e2, temperature_k=300.0,
                                     gravity_m_s2=9.81, mole_fractions={"N2": 1.0}, layers=80)
    expected_top = 8.31446261815324 * 300 / (0.0280134 * 9.81) * np.log(1e5 / 1e2)
    dz = p.altitude_m[1] - p.altitude_m[0]
    assert abs((p.altitude_m[-1] + dz / 2) - expected_top) / expected_top < 2e-3
    assert np.all(np.diff(p.pressure_pa) < 0)
    assert np.all(p.density_kg_m3 > 0)
    assert p.hydrostatic_relative_residual < 4e-4
