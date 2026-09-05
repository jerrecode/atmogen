from __future__ import annotations

import numpy as np

from atmogen import (
    BUILTIN_FLUID_TRANSPORT,
    fluid_transport_properties,
    liquid_mixture_transport_properties,
)


def test_builtin_transport_records_are_positive_and_provenanced():
    assert {"H2O", "CH4", "C2H6", "NH3"} <= set(BUILTIN_FLUID_TRANSPORT)
    for props in BUILTIN_FLUID_TRANSPORT.values():
        assert props.density_kg_m3 > 0
        assert props.dynamic_viscosity_pa_s > 0
        assert props.surface_tension_n_m > 0
        assert props.reference_temperature_k > 0
        assert props.source
        assert props.validity


def test_transport_lookup_is_alias_aware_and_unknown_is_explicit():
    assert fluid_transport_properties("water") == fluid_transport_properties("H2O")
    assert fluid_transport_properties("definitely-not-a-species") is None


def test_mixture_transport_preserves_pure_limits_and_is_bounded():
    water = fluid_transport_properties("H2O")
    methane = fluid_transport_properties("CH4")
    assert water is not None and methane is not None

    pure = liquid_mixture_transport_properties(species_mass_kg={"H2O": 2.0})
    assert pure is not None
    assert np.isclose(pure.density_kg_m3, water.density_kg_m3)
    assert np.isclose(pure.dynamic_viscosity_pa_s, water.dynamic_viscosity_pa_s)
    assert np.isclose(pure.surface_tension_n_m, water.surface_tension_n_m)

    mix = liquid_mixture_transport_properties(
        species_mass_kg={"H2O": 1.0, "CH4": 1.0}
    )
    assert mix is not None
    assert min(water.density_kg_m3, methane.density_kg_m3) <= mix.density_kg_m3 <= max(
        water.density_kg_m3, methane.density_kg_m3
    )
    assert min(water.dynamic_viscosity_pa_s, methane.dynamic_viscosity_pa_s) <= mix.dynamic_viscosity_pa_s <= max(
        water.dynamic_viscosity_pa_s, methane.dynamic_viscosity_pa_s
    )
    assert np.isclose(sum(mix.mass_fractions.values()), 1.0)


def test_mixture_returns_none_when_component_transport_data_are_missing():
    assert liquid_mixture_transport_properties(
        species_mass_kg={"H2O": 1.0, "C_graphite": 1.0}
    ) is None
