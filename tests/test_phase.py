from atmogen import PlanetPhysicalState, SurfaceReservoirs
from atmogen.phase import partition_surface_reservoirs, saturation_pressure_pa


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


def test_worldgen_regression_condensables_have_explicit_models():
    for species, temperature in (("CH4", 94.0), ("C2H6", 94.0), ("SO2", 250.0)):
        pressure, note = saturation_pressure_pa(species, temperature)
        assert pressure is not None and pressure >= 0
        assert note and "estimated" in note
