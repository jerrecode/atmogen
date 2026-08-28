import numpy as np

from atmogen.transport import (eddy_diffusion_flux_mol_m2_s,
                               integrate_eddy_diffusion, mixing_timescale_s,
                               quench_diagnostic)


def test_eddy_flux_has_expected_direction_and_zero_boundary_flux():
    interfaces = np.asarray([0.0, 100.0, 200.0])
    density = np.asarray([10.0, 10.0])
    fraction = np.asarray([1.0, 0.0])
    flux = eddy_diffusion_flux_mol_m2_s(
        altitude_interface_m=interfaces,
        total_molar_density_mol_m3=density,
        mole_fraction=fraction,
        eddy_diffusivity_m2_s=5.0,
    )
    assert flux.shape == (3,)
    assert flux[0] == 0.0 and flux[-1] == 0.0
    assert flux[1] > 0.0  # upward from high lower-layer mixing ratio
    assert np.isclose(flux[1], 0.5)


def test_uniform_profile_is_stationary_under_eddy_diffusion():
    interfaces = np.linspace(0.0, 10_000.0, 21)
    density = np.linspace(40.0, 5.0, 20)
    initial = np.full(20, 0.2)
    result = integrate_eddy_diffusion(
        altitude_interface_m=interfaces,
        total_molar_density_mol_m3=density,
        initial_mole_fractions={"tracer": initial},
        eddy_diffusivity_m2_s=25.0,
        duration_s=2.0e6,
    )
    assert np.allclose(result.final_mole_fraction["tracer"], initial, rtol=0, atol=2e-12)
    assert result.column_conservation_relative < 2e-10


def test_complementary_species_mix_conservatively_and_reduce_variance():
    interfaces = np.linspace(0.0, 20_000.0, 41)
    centers = 0.5 * (interfaces[:-1] + interfaces[1:])
    density = 40.0 * np.exp(-centers / 9000.0) + 2.0
    a = np.where(centers < 10_000.0, 0.95, 0.05)
    b = 1.0 - a
    result = integrate_eddy_diffusion(
        altitude_interface_m=interfaces,
        total_molar_density_mol_m3=density,
        initial_mole_fractions={"A": a, "B": b},
        eddy_diffusivity_m2_s=np.geomspace(8.0, 80.0, 40),
        duration_s=8.0e6,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-12,
    )
    final_a = result.final_mole_fraction["A"]
    final_b = result.final_mole_fraction["B"]
    assert np.var(final_a) < np.var(a)
    assert np.all(final_a >= 0) and np.all(final_b >= 0)
    assert np.allclose(final_a + final_b, 1.0, rtol=0, atol=2e-7)
    assert result.column_conservation_relative < 2e-7
    assert result.nlu > 0


def test_nonuniform_column_conserves_each_tracer_with_zero_flux_boundaries():
    interfaces = np.asarray([0.0, 100.0, 260.0, 600.0, 1200.0, 2200.0])
    density = np.asarray([100.0, 80.0, 45.0, 20.0, 5.0])
    tracer = np.asarray([0.02, 0.04, 0.08, 0.20, 0.60])
    result = integrate_eddy_diffusion(
        altitude_interface_m=interfaces,
        total_molar_density_mol_m3=density,
        initial_mole_fractions={"trace": tracer},
        eddy_diffusivity_m2_s=np.asarray([2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        duration_s=5.0e5,
        method="Radau",
        relative_tolerance=1e-9,
        absolute_tolerance=1e-13,
    )
    initial_column = result.initial_column_moles_m2["trace"]
    final_column = result.final_column_moles_m2["trace"]
    assert np.isclose(final_column, initial_column, rtol=2e-8)
    assert result.column_conservation_relative < 2e-8


def test_mixing_timescale_and_quench_crossing_are_derived_not_hard_coded():
    chemical = np.asarray([10.0, 100.0, 1_000.0, 10_000.0])
    length = np.full(4, 100.0)
    kzz = np.asarray([1000.0, 100.0, 10.0, 1.0])
    mixing = mixing_timescale_s(length, kzz)
    assert np.allclose(mixing, [10.0, 100.0, 1000.0, 10000.0])

    shifted_chemical = np.asarray([1.0, 20.0, 2_000.0, 20_000.0])
    diagnostic = quench_diagnostic(
        chemical_timescale_s=shifted_chemical,
        mixing_length_m=length,
        eddy_diffusivity_m2_s=kzz,
    )
    assert diagnostic.quenched.tolist() == [False, False, True, True]
    assert diagnostic.crossing_indices == (2,)
