import pytest

from atmogen import (
    API_SCHEMA_VERSION,
    BUILTIN_DATABASE,
    ColumnBatchInput,
    ColumnInput,
    ColumnSurfaceBoundary,
    ElementInventory,
    PlanetPhysicalState,
    SolverSettings,
    SurfaceBoundaryMode,
    blackbody_stellar_spectrum,
    column_state_fingerprint,
    pressure_from_elevation,
    solve_columns,
    solve_columns_with_diagnostics,
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


def test_public_column_fingerprint_is_stable_and_versioned():
    inventory_a = ElementInventory(
        {"N": 2.0, "O": 1.0},
        {"N2": 1.0, "O2": 0.5},
        "fingerprint-order-regression",
    )
    inventory_b = ElementInventory(
        {"O": 1.0, "N": 2.0},
        {"O2": 0.5, "N2": 1.0},
        "fingerprint-order-regression",
    )
    planet = PlanetPhysicalState(6.371e6, 9.80665, 101325.0)
    star = blackbody_stellar_spectrum(5772.0, 1361.0)
    settings = SolverSettings(chemistry_mode="fixed_species", vertical_layers=8)
    first = column_state_fingerprint(ColumnInput(planet, inventory_a), star, settings)
    second = column_state_fingerprint(ColumnInput(planet, inventory_b), star, settings)
    assert first == second
    assert len(first) == 64
    assert API_SCHEMA_VERSION == 12
    assert len(BUILTIN_DATABASE.revision_hash) == 64

    changed_setting = column_state_fingerprint(
        ColumnInput(planet, inventory_a),
        star,
        SolverSettings(chemistry_mode="fixed_species", vertical_layers=9),
    )
    changed_forcing = column_state_fingerprint(
        ColumnInput(planet, inventory_a, stellar_flux_scale=0.9), star, settings
    )
    assert changed_setting != first
    assert changed_forcing != first


def test_detailed_batch_diagnostics_are_order_independent_and_state_aligned():
    inventory = ElementInventory(
        species_moles_to_elements({"N2": 0.999, "CO2": 0.001}),
        {"N2": 0.999, "CO2": 0.001},
        "batch-diagnostic-regression",
    )
    planet = PlanetPhysicalState(6.371e6, 9.80665, 101325.0, 280.0, 0.2)
    low = ColumnInput(planet, inventory, stellar_flux_scale=0.65)
    high = ColumnInput(planet, inventory, stellar_flux_scale=1.35)
    star = blackbody_stellar_spectrum(5772.0, 1361.0)
    settings = SolverSettings(chemistry_mode="fixed_species", vertical_layers=8)

    first = solve_columns_with_diagnostics(
        ColumnBatchInput((low, high, low), star), settings
    )
    reordered = solve_columns_with_diagnostics(
        ColumnBatchInput((high, low, low), star), settings
    )
    diagnostics = first.diagnostics
    assert diagnostics.input_count == 3
    assert diagnostics.unique_state_count == 2
    assert diagnostics.deduplicated_count == 1
    assert diagnostics.deduplication_ratio == 1 / 3
    assert diagnostics.reused == (False, False, True)
    assert diagnostics.fingerprints[0] == diagnostics.fingerprints[2]
    assert diagnostics.unique_state_index[0] == diagnostics.unique_state_index[2]
    assert diagnostics.unique_fingerprints == reordered.diagnostics.unique_fingerprints
    assert diagnostics.database_sha256 == BUILTIN_DATABASE.revision_hash
    assert diagnostics.converged_count == sum(
        result.convergence.converged for result in first.results
    )
    expected_fallbacks = [
        tuple(result.diagnostics.get("fallbacks", ())) for result in first.results
    ]
    assert diagnostics.fallback_column_count == sum(bool(x) for x in expected_fallbacks)
    assert diagnostics.fallback_event_count == sum(len(x) for x in expected_fallbacks)
    assert len(diagnostics.per_column_provenance) == 3
    assert first.results[0] is first.results[2]
    assert reordered.results[1] is reordered.results[2]
    assert (
        first.results[0].atmosphere.temperature_k[0]
        == reordered.results[1].atmosphere.temperature_k[0]
    )
    assert (
        first.results[1].atmosphere.temperature_k[0]
        == reordered.results[0].atmosphere.temperature_k[0]
    )


def test_hydrostatic_surface_pressure_boundary_tracks_highlands_and_basins():
    parent_pressure = 101325.0
    kwargs = {
        "gravity_m_s2": 9.80665,
        "reference_temperature_k": 280.0,
        "mean_molar_mass_kg_mol": 0.02897,
    }
    mountain = pressure_from_elevation(parent_pressure, 2500.0, **kwargs)
    basin = pressure_from_elevation(parent_pressure, -500.0, **kwargs)
    assert mountain < parent_pressure < basin
    assert pressure_from_elevation(parent_pressure, 0.0, **kwargs) == parent_pressure


def test_adjusted_boundary_is_validated_and_reported_by_batch():
    species = {"N2": 0.79, "O2": 0.21}
    inventory = ElementInventory(
        species_moles_to_elements(species), species, "boundary-regression"
    )
    parent_pressure = 101325.0
    elevation = 1800.0
    temperature = 278.0
    molar_mass = 0.02885
    pressure = pressure_from_elevation(
        parent_pressure,
        elevation,
        gravity_m_s2=9.80665,
        reference_temperature_k=temperature,
        mean_molar_mass_kg_mol=molar_mass,
    )
    boundary = ColumnSurfaceBoundary(
        mode=SurfaceBoundaryMode.HYDROSTATIC_ADJUSTED,
        elevation_delta_m=elevation,
        parent_surface_pressure_pa=parent_pressure,
        reference_temperature_k=temperature,
        mean_molar_mass_kg_mol=molar_mass,
    )
    column = ColumnInput(
        PlanetPhysicalState(6.371e6, 9.80665, pressure, temperature, 0.2),
        inventory,
        surface_boundary=boundary,
    )
    result = solve_columns_with_diagnostics(
        ColumnBatchInput(
            (column,), blackbody_stellar_spectrum(5772.0, 1361.0)
        ),
        SolverSettings(chemistry_mode="fixed_species", vertical_layers=8),
    )
    assert result.diagnostics.surface_boundary_modes == (
        "hydrostatic_adjusted",
    )
    recorded = result.diagnostics.surface_boundaries[0]
    assert recorded["surface_pressure_pa"] == pressure
    assert recorded["parent_surface_pressure_pa"] == parent_pressure
    assert recorded["elevation_delta_m"] == elevation
    assert result.results[0].surface.mass_closure_relative < 1e-10

    with pytest.raises(ValueError, match="does not match"):
        ColumnInput(
            PlanetPhysicalState(6.371e6, 9.80665, parent_pressure, temperature),
            inventory,
            surface_boundary=boundary,
        )
