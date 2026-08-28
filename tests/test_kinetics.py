import numpy as np
import pytest

from atmogen import BUILTIN_DATABASE
from atmogen.database import ChemicalDatabase, ProvenanceClass, Reaction
from atmogen.kinetics import (arrhenius_rate_constant, expected_rate_coefficient_units,
                              integrate_kinetics, reaction_rates)


def co2_dissociation(rate_s1: float = 0.2) -> Reaction:
    return Reaction(
        key="co2_dissociation_test",
        reactants={"CO2": 1.0},
        products={"CO": 1.0, "O2": 0.5},
        rate_law="arrhenius",
        pre_exponential_factor_si=rate_s1,
        rate_coefficient_units="s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic analytical regression reaction",
        validity="test only; constant first-order coefficient",
    )


def test_first_order_arrhenius_network_matches_analytic_solution_and_closes_elements():
    reaction = co2_dissociation(0.2)
    result = integrate_kinetics(
        reactions=(reaction,),
        initial_concentration_mol_m3={"CO2": 2.0},
        temperature_k=300.0,
        duration_s=10.0,
        relative_tolerance=1e-10,
        absolute_tolerance_mol_m3=1e-13,
    )
    expected_co2 = 2.0 * np.exp(-2.0)
    consumed = 2.0 - expected_co2
    assert np.isclose(result.final_concentration_mol_m3["CO2"], expected_co2, rtol=2e-8)
    assert np.isclose(result.final_concentration_mol_m3["CO"], consumed, rtol=2e-8)
    assert np.isclose(result.final_concentration_mol_m3["O2"], 0.5 * consumed, rtol=2e-8)
    assert result.element_relative_residual < 2e-9
    assert result.converged
    assert result.method == "BDF"
    assert result.nfev > 0


def test_arrhenius_temperature_and_order_semantics_are_explicit():
    reaction = Reaction(
        key="water_formation_test",
        reactants={"H2": 1.0, "O2": 0.5},
        products={"H2O": 1.0},
        rate_law="arrhenius",
        pre_exponential_factor_si=3.0,
        temperature_exponent=1.0,
        activation_energy_j_mol=1000.0,
        reference_temperature_k=300.0,
        rate_coefficient_units="(m^3 mol^-1)^0.5 s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic unit regression reaction",
        validity="test only",
    )
    assert np.isclose(reaction.order, 1.5)
    assert expected_rate_coefficient_units(reaction.order) == "(m^3 mol^-1)^0.5 s^-1"
    k300 = arrhenius_rate_constant(reaction, 300.0)
    k600 = arrhenius_rate_constant(reaction, 600.0)
    assert k600 > k300 > 0
    rates = reaction_rates(
        reactions=(reaction,),
        concentration_mol_m3={"H2": 4.0, "O2": 9.0},
        temperature_k=300.0,
    )
    assert np.isclose(rates[reaction.key], k300 * 4.0 * 9.0**0.5)


def test_stiff_reversible_network_remains_nonnegative_and_element_conserving():
    forward = Reaction(
        key="water_fast_forward_test",
        reactants={"H2": 1.0, "O2": 0.5},
        products={"H2O": 1.0},
        rate_law="arrhenius",
        pre_exponential_factor_si=2.0e4,
        rate_coefficient_units="(m^3 mol^-1)^0.5 s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic stiff regression network",
        validity="test only",
    )
    reverse = Reaction(
        key="water_slow_reverse_test",
        reactants={"H2O": 1.0},
        products={"H2": 1.0, "O2": 0.5},
        rate_law="arrhenius",
        pre_exponential_factor_si=2.0,
        rate_coefficient_units="s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic stiff regression network",
        validity="test only",
    )
    result = integrate_kinetics(
        reactions=(forward, reverse),
        initial_concentration_mol_m3={"H2": 2.0, "O2": 1.0, "H2O": 1e-12},
        temperature_k=500.0,
        duration_s=1.0,
        method="BDF",
        relative_tolerance=1e-8,
        absolute_tolerance_mol_m3=1e-12,
    )
    assert result.element_relative_residual < 2e-7
    assert min(result.final_concentration_mol_m3.values()) >= 0
    assert result.nlu > 0
    assert result.nfev > 0


def test_database_rejects_unbalanced_reaction_and_hashes_reaction_data():
    bad = Reaction(
        key="bad_unbalanced_test",
        reactants={"CO2": 1.0},
        products={"CO": 1.0},
        rate_law="arrhenius",
        pre_exponential_factor_si=1.0,
        rate_coefficient_units="s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="deliberately invalid regression reaction",
        validity="test only",
    )
    with pytest.raises(ValueError, match="not element-balanced"):
        ChemicalDatabase(BUILTIN_DATABASE.species, reactions=(bad,))

    a = ChemicalDatabase(BUILTIN_DATABASE.species, reactions=(co2_dissociation(0.1),))
    b = ChemicalDatabase(BUILTIN_DATABASE.species, reactions=(co2_dissociation(0.2),))
    assert a.revision_hash != b.revision_hash


def test_photolysis_reaction_requires_external_j_rate():
    reaction = Reaction(
        key="co2_photolysis_test",
        reactants={"CO2": 1.0},
        products={"CO": 1.0, "O2": 0.5},
        rate_law="photolysis",
        rate_coefficient_units="s^-1",
        provenance_class=ProvenanceClass.ESTIMATED,
        source="synthetic photolysis regression reaction",
        validity="test only",
    )
    with pytest.raises(KeyError, match="missing photolysis rate"):
        reaction_rates(
            reactions=(reaction,),
            concentration_mol_m3={"CO2": 1.0},
            temperature_k=300.0,
        )
