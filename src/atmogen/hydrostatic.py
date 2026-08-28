from __future__ import annotations

from typing import Mapping

import numpy as np

from .chemistry import R_GAS
from .database import BUILTIN_DATABASE, ChemicalDatabase
from .models import AtmosphericProfile


def logarithmic_pressure_interfaces(surface_pressure_pa: float, top_pressure_pa: float, layers: int) -> np.ndarray:
    if surface_pressure_pa <= top_pressure_pa > 0 or top_pressure_pa <= 0:
        raise ValueError("require surface_pressure_pa > top_pressure_pa > 0")
    if layers < 1:
        raise ValueError("layers must be positive")
    return np.geomspace(surface_pressure_pa, top_pressure_pa, layers + 1, dtype=np.float64)


def mean_molar_mass(mole_fractions: Mapping[str, float], database: ChemicalDatabase = BUILTIN_DATABASE) -> float:
    total = sum(max(float(v), 0.0) for v in mole_fractions.values())
    if total <= 0:
        raise ValueError("mole fractions must have positive total")
    return float(sum(max(float(v), 0.0) * database.get(k).molar_mass_kg_mol for k, v in mole_fractions.items()) / total)


def solve_isothermal_hydrostatic(*, surface_pressure_pa: float, top_pressure_pa: float, temperature_k: float,
                                 gravity_m_s2: float, mole_fractions: Mapping[str, float], layers: int,
                                 database: ChemicalDatabase = BUILTIN_DATABASE) -> AtmosphericProfile:
    if temperature_k <= 0 or gravity_m_s2 <= 0:
        raise ValueError("temperature and gravity must be positive")
    pi = logarithmic_pressure_interfaces(surface_pressure_pa, top_pressure_pa, layers)
    # Logarithmic-cell mean pressure makes the finite-volume hydrostatic balance
    # exact for an isothermal exponential atmosphere, unlike a geometric midpoint.
    p = (pi[:-1] - pi[1:]) / np.log(pi[:-1] / pi[1:])
    molar_mass = mean_molar_mass(mole_fractions, database)
    scale_height = R_GAS * temperature_k / (molar_mass * gravity_m_s2)
    zi = scale_height * np.log(surface_pressure_pa / pi)
    z = 0.5 * (zi[:-1] + zi[1:])
    rho = p * molar_mass / (R_GAS * temperature_k)
    # Integral-cell residual: delta P + integral(rho*g dz).
    layer_mass_pressure = pi[:-1] - pi[1:]
    reconstructed = rho * gravity_m_s2 * (zi[1:] - zi[:-1])
    residual = float(np.max(np.abs(reconstructed - layer_mass_pressure) / np.maximum(layer_mass_pressure, 1e-30)))
    return AtmosphericProfile(p, pi, z, np.full(layers, temperature_k), rho, molar_mass,
                              dict(mole_fractions), residual)
