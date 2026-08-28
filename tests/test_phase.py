from atmogen import PlanetPhysicalState, SurfaceReservoirs
from atmogen.phase import (atmospheric_composition_with_surface_vapor,
                           partition_surface_reservoirs, saturation_pressure_pa)


def test_iapws_water_saturation_near_one_atmosphere_at_boiling():
    pressure, fallback = saturation_pressure_pa("H2O", 373.1243)
    assert fallback is None
    assert abs(pressure - 101325.0) / 101325.0 < 0.003


def test_partition_conserves_surface_reservoir_mass():
    result = partition_surface_reservoirs(planet=PlanetPhysicalState(6.371e6, 9.81, 1e5), temperature_k=288.15,
                                          atmospheric_mole_fractions={"N2": 0.99, "H2O": 0.01},
                                          surface=SurfaceReservoirs({"H2O": 1e18}))
    assert result.mass_closure_relative < 1e-14
    assert result.liquid_mass_kg["H2O"] > 0
    assert result.liquid_volume_m3["H2O"] > 0
    assert result.surface_vapor_mole_fractions["H2O"] > 0
    assert len(result.liquid_phases) == 1


def test_worldgen_regression_condensables_have_explicit_models():
    for species, temperature in (("CH4", 94.0), ("C2H6", 94.0), ("SO2", 250.0)):
        pressure, note = saturation_pressure_pa(species, temperature)
        assert pressure is not None and pressure >= 0
        assert note and "estimated" in note


def test_titan_like_multicomponent_reservoir_uses_explicit_ideal_fallback():
    result = partition_surface_reservoirs(
        planet=PlanetPhysicalState(2.575e6, 1.352, 1.47e5, 94.0),
        temperature_k=94.0,
        atmospheric_mole_fractions={"N2": 0.95, "CH4": 0.05},
        surface=SurfaceReservoirs({"CH4": 5.0e19, "C2H6": 5.0e19}),
        activity_model="auto",
    )
    assert result.mass_closure_relative < 2e-12
    assert result.liquid_mass_kg["CH4"] > 0
    assert result.liquid_mass_kg["C2H6"] > 0
    assert result.activity_model == "ideal"
    assert len(result.liquid_phases) == 1
    assert abs(sum(result.liquid_phases[0].mole_fractions.values()) - 1.0) < 1e-12
    assert any("used ideal" in note for note in result.fallbacks)
    assert sum(result.surface_vapor_mole_fractions.values()) > 0


def test_surface_vapor_is_part_of_fixed_pressure_atmospheric_composition():
    planet = PlanetPhysicalState(6.371e6, 9.80665, 101325.0)
    composition, source = atmospheric_composition_with_surface_vapor(
        planet=planet,
        atmospheric_mole_fractions={"N2": 0.79, "O2": 0.21},
        surface_vapor_mass_kg={"H2O": 1.0e16},
    )
    assert abs(sum(composition.values()) - 1.0) < 1e-12
    assert source["H2O"] > 0
    assert composition["H2O"] >= source["H2O"]
    assert composition["N2"] < 0.79
