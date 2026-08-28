from atmogen import (
    ColumnBatchInput,
    ColumnInput,
    ElementInventory,
    PlanetPhysicalState,
    SolverSettings,
    blackbody_stellar_spectrum,
    solve_columns,
    species_moles_to_elements,
)


def test_batch_column_flux_scale_changes_equilibrium_temperature():
    species = {"N2": 0.999, "CO2": 0.001}
    inventory = ElementInventory(
        species_moles_to_elements(species), species, "column-forcing-regression"
    )
    planet = PlanetPhysicalState(6.371e6, 9.80665, 101325.0, 280.0, 0.2)
    low = ColumnInput(planet, inventory, stellar_flux_scale=0.65)
    high = ColumnInput(planet, inventory, stellar_flux_scale=1.35)
    results = solve_columns(
        ColumnBatchInput((low, high), blackbody_stellar_spectrum(5772.0, 1361.0)),
        SolverSettings(chemistry_mode="fixed_species", vertical_layers=8),
    )
    assert results[1].atmosphere.temperature_k[0] > results[0].atmosphere.temperature_k[0]
    assert results[0].provenance["stellar_spectrum"].endswith("column stellar flux scale=0.65")
    assert results[1].provenance["stellar_spectrum"].endswith("column stellar flux scale=1.35")


def test_equal_forcing_columns_still_deduplicate():
    species = {"N2": 1.0}
    inventory = ElementInventory(
        species_moles_to_elements(species), species, "dedupe-forcing-regression"
    )
    planet = PlanetPhysicalState(6.371e6, 9.80665, 101325.0)
    column = ColumnInput(planet, inventory, stellar_flux_scale=0.9)
    results = solve_columns(
        ColumnBatchInput((column, column), blackbody_stellar_spectrum(5772.0, 1361.0)),
        SolverSettings(chemistry_mode="fixed_species", vertical_layers=8),
    )
    assert results[0] is results[1]
