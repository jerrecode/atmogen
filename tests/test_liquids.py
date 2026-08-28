from __future__ import annotations

import numpy as np

from atmogen.database import NRTLInteraction, ProvenanceClass
from atmogen.liquids import (IdealActivityModel, NRTLActivityModel,
                             liquid_phase_stability, select_activity_model)
from atmogen import BUILTIN_DATABASE


def test_ideal_activity_coefficients_are_unity_and_gibbs_is_convex():
    model = IdealActivityModel()
    gamma = model.activity_coefficients(
        temperature_k=300.0, mole_fractions={"A": 0.25, "B": 0.75}
    )
    assert gamma == {"A": 1.0, "B": 1.0}
    split = liquid_phase_stability(
        temperature_k=300.0,
        mole_fractions={"A": 0.5, "B": 0.5},
        activity_model=model,
    )
    assert split.single_phase_stable
    assert len(split.phase_compositions) == 1
    assert np.isclose(sum(split.phase_compositions[0].values()), 1.0)


def test_nrtl_strong_repulsion_detects_binary_liquid_liquid_split():
    interactions = (
        NRTLInteraction(
            "A", "B", 12000.0, 0.3, ProvenanceClass.ESTIMATED,
            "synthetic regression parameter", "test-only strong positive deviation",
        ),
        NRTLInteraction(
            "B", "A", 12000.0, 0.3, ProvenanceClass.ESTIMATED,
            "synthetic regression parameter", "test-only strong positive deviation",
        ),
    )
    model = NRTLActivityModel(interactions)
    gamma = model.activity_coefficients(
        temperature_k=300.0, mole_fractions={"A": 0.5, "B": 0.5}
    )
    assert gamma["A"] > 1.0
    assert gamma["B"] > 1.0
    split = liquid_phase_stability(
        temperature_k=300.0,
        mole_fractions={"A": 0.5, "B": 0.5},
        activity_model=model,
    )
    assert not split.single_phase_stable
    assert len(split.phase_compositions) == 2
    x_a = split.phase_compositions[0]["A"]
    x_b = split.phase_compositions[1]["A"]
    assert x_a < 0.1
    assert x_b > 0.9
    reconstructed = (
        split.phase_fractions_mol[0] * x_a
        + split.phase_fractions_mol[1] * x_b
    )
    assert np.isclose(reconstructed, 0.5, atol=1e-10)
    assert split.split_reduced_gibbs < split.single_phase_reduced_gibbs


def test_auto_activity_selection_never_invents_missing_parameters():
    model, fallbacks = select_activity_model(
        species=("CH4", "C2H6"), database=BUILTIN_DATABASE, mode="auto"
    )
    assert model.name == "ideal"
    assert fallbacks
    assert "used ideal" in fallbacks[0]
