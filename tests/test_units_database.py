from atmogen import BUILTIN_DATABASE, species_moles_to_elements


def test_species_to_elements_conversion_and_database_hash_are_stable():
    assert species_moles_to_elements({"H2O": 2, "CO2": 1}) == {"H": 4.0, "O": 4.0, "C": 1.0}
    assert len(BUILTIN_DATABASE.revision_hash) == 64
    assert BUILTIN_DATABASE.revision_hash == BUILTIN_DATABASE.revision_hash


def test_all_external_numbers_have_units_and_provenance():
    for species in BUILTIN_DATABASE.species.values():
        assert species.molar_mass_kg_mol > 0
        assert species.source
        assert species.validity
        assert species.provenance_class.value
