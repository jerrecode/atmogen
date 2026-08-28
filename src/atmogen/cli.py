from __future__ import annotations

import argparse
import json

from . import (ElementInventory, Fidelity, PlanetPhysicalState, SolverSettings, SurfaceReservoirs,
               blackbody_stellar_spectrum, solve_planet, species_moles_to_elements)


def earth_demo() -> dict[str, object]:
    species = {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}
    inventory = ElementInventory(species_moles_to_elements(species), species, "legacy molecular initial state")
    result = solve_planet(planet=PlanetPhysicalState(6_371_000.0, 9.80665, 101325.0),
                          star=blackbody_stellar_spectrum(5772.0, 1361.0), inventory=inventory,
                          surface=SurfaceReservoirs({"H2O": 1.4e21}),
                          settings=SolverSettings(fidelity=Fidelity.FAST, chemistry_mode="fixed_species"))
    return {"surface_temperature_k": float(result.atmosphere.temperature_k[0]),
            "bond_albedo": result.spectra.bond_albedo, "visible_srgb": result.spectra.visible_srgb,
            "energy_imbalance_w_m2": result.energy_budget.imbalance_w_m2,
            "converged": result.convergence.converged, "provenance": result.provenance,
            "diagnostics": result.diagnostics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a compact standalone atmogen validation column")
    parser.add_argument("--demo", choices=["earth"], default="earth")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(earth_demo(), indent=None if args.compact else 2, default=list, sort_keys=True))
    return 0
