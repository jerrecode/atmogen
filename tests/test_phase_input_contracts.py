import numpy as np
import pytest

from atmogen.models import PlanetPhysicalState, SolverSettings
from atmogen.phase import (
    atmospheric_composition_with_surface_vapor,
    partition_surface_reservoirs,
)
from atmogen.models import SurfaceReservoirs


def _planet():
    return PlanetPhysicalState(
        radius_m=6.371e6,
        gravity_m_s2=9.81,
        surface_pressure_pa=1.0e5,
    )


@pytest.mark.parametrize("layers", [4.5, np.nan, np.inf, True])
def test_solver_settings_vertical_layers_must_be_integer(layers):
    with pytest.raises((TypeError, ValueError), match="vertical_layers.*integer"):
        SolverSettings(vertical_layers=layers)


@pytest.mark.parametrize(
    "background",
    [
        {"N2": np.nan, "O2": 1.0},
        {"N2": np.inf, "O2": 1.0},
        {"N2": -0.1, "O2": 1.1},
    ],
)
def test_surface_vapor_composition_rejects_invalid_background(background):
    with pytest.raises(ValueError, match="atmospheric_mole_fractions.*finite and non-negative"):
        atmospheric_composition_with_surface_vapor(
            planet=_planet(),
            atmospheric_mole_fractions=background,
            surface_vapor_mass_kg={},
        )


@pytest.mark.parametrize("mass", [np.nan, np.inf, -1.0])
def test_surface_vapor_composition_rejects_invalid_direct_vapor_mass(mass):
    with pytest.raises(ValueError, match="surface vapor mass.*finite and non-negative"):
        atmospheric_composition_with_surface_vapor(
            planet=_planet(),
            atmospheric_mole_fractions={"N2": 1.0},
            surface_vapor_mass_kg={"H2O": mass},
        )


@pytest.mark.parametrize("temperature", [np.nan, np.inf, -np.inf, 0.0])
def test_surface_partition_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature_k.*finite and positive"):
        partition_surface_reservoirs(
            planet=_planet(),
            temperature_k=temperature,
            atmospheric_mole_fractions={"N2": 1.0},
            surface=SurfaceReservoirs({}),
        )
