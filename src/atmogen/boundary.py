from __future__ import annotations

import numpy as np


MOLAR_GAS_CONSTANT_J_MOL_K = 8.31446261815324


def pressure_from_elevation(
    parent_surface_pressure_pa: float,
    elevation_delta_m: float,
    *,
    gravity_m_s2: float,
    reference_temperature_k: float,
    mean_molar_mass_kg_mol: float,
) -> float:
    """Adjust a parent pressure with an isothermal hydrostatic column.

    ``elevation_delta_m`` is positive upward from the parent datum.  This helper
    supplies a documented host-model boundary estimate; it does not reinterpret
    the vertical coordinate of the subsequently solved atmospheric profile.
    """
    values = {
        "parent_surface_pressure_pa": parent_surface_pressure_pa,
        "gravity_m_s2": gravity_m_s2,
        "reference_temperature_k": reference_temperature_k,
        "mean_molar_mass_kg_mol": mean_molar_mass_kg_mol,
    }
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(elevation_delta_m):
        raise ValueError("elevation_delta_m must be finite")
    exponent = (
        -float(mean_molar_mass_kg_mol)
        * float(gravity_m_s2)
        * float(elevation_delta_m)
        / (MOLAR_GAS_CONSTANT_J_MOL_K * float(reference_temperature_k))
    )
    pressure = float(parent_surface_pressure_pa) * float(np.exp(exponent))
    if not np.isfinite(pressure) or pressure <= 0:
        raise ValueError("elevation adjustment produced a non-finite pressure")
    return pressure
