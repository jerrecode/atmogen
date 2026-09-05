from __future__ import annotations

import numpy as np

from atmogen import (
    BUILTIN_FLUID_TRANSPORT,
    fluid_transport_properties,
    liquid_mixture_transport_fields,
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


def test_vectorized_transport_fields_preserve_pure_limits_and_inactive_mask():
    water = fluid_transport_properties("H2O")
    assert water is not None
    fields = liquid_mixture_transport_fields(
        species_mass_kg={
            "H2O": np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float),
        }
    )
    assert fields is not None
    active = np.array([[True, False], [True, False]])
    assert np.array_equal(fields.active_mask, active)
    assert np.allclose(fields.density_kg_m3[active], water.density_kg_m3)
    assert np.allclose(
        fields.dynamic_viscosity_pa_s[active], water.dynamic_viscosity_pa_s
    )
    assert np.allclose(fields.surface_tension_n_m[active], water.surface_tension_n_m)
    assert np.count_nonzero(fields.density_kg_m3[~active]) == 0
    assert np.count_nonzero(fields.dynamic_viscosity_pa_s[~active]) == 0
    assert np.count_nonzero(fields.surface_tension_n_m[~active]) == 0


def test_vectorized_transport_fields_match_scalar_mixture_cellwise():
    water = fluid_transport_properties("H2O")
    methane = fluid_transport_properties("CH4")
    assert water is not None and methane is not None
    fields = liquid_mixture_transport_fields(
        species_mass_kg={
            "H2O": np.array([[1.0, 0.0, 1.0, 0.0]]),
            "CH4": np.array([[0.0, 1.0, 1.0, 0.0]]),
        }
    )
    assert fields is not None
    assert np.isclose(fields.density_kg_m3[0, 0], water.density_kg_m3)
    assert np.isclose(fields.density_kg_m3[0, 1], methane.density_kg_m3)

    scalar = liquid_mixture_transport_properties(
        species_mass_kg={"H2O": 1.0, "CH4": 1.0}
    )
    assert scalar is not None
    assert np.isclose(fields.density_kg_m3[0, 2], scalar.density_kg_m3)
    assert np.isclose(
        fields.dynamic_viscosity_pa_s[0, 2], scalar.dynamic_viscosity_pa_s
    )
    assert np.isclose(fields.surface_tension_n_m[0, 2], scalar.surface_tension_n_m)

    summed = sum(fields.mass_fractions.values())
    assert np.allclose(summed[fields.active_mask], 1.0)
    assert np.count_nonzero(summed[~fields.active_mask]) == 0


def test_vectorized_transport_fields_coalesce_aliases_and_validate_contracts():
    aliased = liquid_mixture_transport_fields(
        species_mass_kg={
            "water": np.array([1.0, 0.0]),
            "H2O": np.array([2.0, 0.0]),
        }
    )
    assert aliased is not None
    assert set(aliased.mass_fractions) == {"H2O"}
    assert np.isclose(aliased.total_mass_kg[0], 3.0)

    assert liquid_mixture_transport_fields(
        species_mass_kg={"H2O": np.zeros(3)}
    ) is None
    assert liquid_mixture_transport_fields(
        species_mass_kg={"H2O": np.ones(2), "C_graphite": np.ones(2)}
    ) is None

    import pytest

    with pytest.raises(ValueError, match="one shape"):
        liquid_mixture_transport_fields(
            species_mass_kg={"H2O": np.ones(2), "CH4": np.ones(3)}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        liquid_mixture_transport_fields(
            species_mass_kg={"H2O": np.array([1.0, np.nan])}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        liquid_mixture_transport_fields(
            species_mass_kg={"H2O": np.array([1.0, -1.0])}
        )
