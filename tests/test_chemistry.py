import numpy as np

from atmogen.chemistry import IdealGibbsEquilibrium


def test_equilibrium_is_nonnegative_and_element_conserving():
    result = IdealGibbsEquilibrium().solve(temperature_k=900.0, pressure_pa=1e5,
                                           element_moles={"H": 2.0, "O": 1.0},
                                           initial_species_moles={"H2": 1.0, "O2": 0.5})
    assert result.converged, result.message
    assert result.element_relative_residual < 2e-8
    assert all(value >= 0 for value in result.species_moles.values())
    assert np.isclose(sum(result.gas_mole_fractions.values()), 1.0)


def test_candidate_pruning_excludes_species_with_missing_elements():
    result = IdealGibbsEquilibrium().solve(temperature_k=300.0, pressure_pa=1e5,
                                           element_moles={"N": 2.0}, initial_species_moles={"N2": 1.0})
    assert result.converged
    assert result.gas_mole_fractions == {"N2": 1.0}
    assert "CO2" in result.pruned_species
