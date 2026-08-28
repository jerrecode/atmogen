import numpy as np

from atmogen import (ColumnBatchInput, ColumnInput, ElementInventory, Fidelity, PlanetPhysicalState,
                     SolverSettings, SurfaceReservoirs, blackbody_stellar_spectrum, solve_columns,
                     solve_planet, species_moles_to_elements)


def earth_case():
    species = {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}
    return dict(planet=PlanetPhysicalState(6_371_000, 9.80665, 101325.0),
                star=blackbody_stellar_spectrum(5772.0, 1361.0),
                inventory=ElementInventory(species_moles_to_elements(species), species, "legacy molecular initial state"),
                surface=SurfaceReservoirs({"H2O": 1.4e21}),
                settings=SolverSettings(fidelity=Fidelity.FAST, chemistry_mode="fixed_species"))


def test_complete_column_is_finite_closed_converged_and_deterministic():
    a = solve_planet(**earth_case())
    b = solve_planet(**earth_case())
    assert a.convergence.converged
    assert a.diagnostics["finite"] and a.diagnostics["non_negative"]
    assert a.surface.mass_closure_relative < 1e-12
    assert abs(a.energy_budget.imbalance_w_m2) < 1e-9
    assert 0 <= a.spectra.bond_albedo < 1
    assert np.array_equal(a.atmosphere.pressure_pa, b.atmosphere.pressure_pa)
    assert a.spectra.visible_srgb == b.spectra.visible_srgb


def test_batch_reuses_identical_column_result_object():
    case = earth_case()
    column = ColumnInput(case["planet"], case["inventory"], case["surface"])
    results = solve_columns(ColumnBatchInput((column, column), case["star"]), case["settings"])
    assert results[0] is results[1]


def test_dense_co2_column_has_stronger_fast_greenhouse_than_thin_co2():
    star = blackbody_stellar_spectrum(5772, 590.0)
    inv = ElementInventory(species_moles_to_elements({"CO2": 1.0}), {"CO2": 1.0}, "specified molecule")
    settings = SolverSettings(chemistry_mode="fixed_species")
    thin = solve_planet(planet=PlanetPhysicalState(3.39e6, 3.71, 600.0, 215.0), star=star,
                        inventory=inv, settings=settings)
    dense = solve_planet(planet=PlanetPhysicalState(6.05e6, 8.87, 9.2e6, 735.0), star=star,
                         inventory=inv, settings=settings)
    assert dense.energy_budget.longwave_optical_depth > thin.energy_budget.longwave_optical_depth
    assert dense.atmosphere.temperature_k[0] > thin.atmosphere.temperature_k[0]
