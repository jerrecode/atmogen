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


def logarithmic_cell_mean_pressure(pressure_interface_pa: np.ndarray) -> np.ndarray:
    interfaces = np.asarray(pressure_interface_pa, dtype=float)
    if interfaces.ndim != 1 or interfaces.size < 2 or np.any(~np.isfinite(interfaces)):
        raise ValueError("pressure interfaces must be a finite 1-D array")
    if np.any(interfaces <= 0) or np.any(np.diff(interfaces) >= 0):
        raise ValueError("pressure interfaces must be positive and strictly decreasing")
    return (interfaces[:-1] - interfaces[1:]) / np.log(interfaces[:-1] / interfaces[1:])


def mean_molar_mass(mole_fractions: Mapping[str, float], database: ChemicalDatabase = BUILTIN_DATABASE) -> float:
    total = sum(max(float(v), 0.0) for v in mole_fractions.values())
    if total <= 0:
        raise ValueError("mole fractions must have positive total")
    return float(sum(max(float(v), 0.0) * database.get(k).molar_mass_kg_mol for k, v in mole_fractions.items()) / total)


def solve_temperature_profile_hydrostatic(
    *,
    pressure_interface_pa: np.ndarray,
    temperature_k: np.ndarray,
    gravity_m_s2: float,
    mole_fractions: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> AtmosphericProfile:
    """Finite-volume ideal-gas hydrostatics for an arbitrary layer temperature profile.

    Layer thickness is chosen from ΔP=ρgΔz using logarithmic-cell-mean pressure.
    This makes discrete column mass/hydrostatic closure exact to floating-point
    precision while allowing the thermal profile to vary by layer.
    """
    pi = np.asarray(pressure_interface_pa, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be finite and positive")
    p = logarithmic_cell_mean_pressure(pi)
    if temperature.shape != p.shape or temperature.ndim != 1:
        raise ValueError("temperature_k must contain one value per pressure layer")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("temperature_k must be finite and positive")
    molar_mass = mean_molar_mass(mole_fractions, database)
    rho = p * molar_mass / (R_GAS * temperature)
    pressure_drop = pi[:-1] - pi[1:]
    dz = pressure_drop / np.maximum(rho * gravity_m_s2, 1e-300)
    zi = np.concatenate(([0.0], np.cumsum(dz)))
    z = 0.5 * (zi[:-1] + zi[1:])
    reconstructed = rho * gravity_m_s2 * dz
    residual = float(np.max(np.abs(reconstructed - pressure_drop) / np.maximum(pressure_drop, 1e-30)))
    return AtmosphericProfile(
        p, pi, z, temperature.copy(), rho, molar_mass, dict(mole_fractions), residual
    )


def solve_isothermal_hydrostatic(*, surface_pressure_pa: float, top_pressure_pa: float, temperature_k: float,
                                 gravity_m_s2: float, mole_fractions: Mapping[str, float], layers: int,
                                 database: ChemicalDatabase = BUILTIN_DATABASE) -> AtmosphericProfile:
    if temperature_k <= 0 or gravity_m_s2 <= 0:
        raise ValueError("temperature and gravity must be positive")
    pi = logarithmic_pressure_interfaces(surface_pressure_pa, top_pressure_pa, layers)
    p = logarithmic_cell_mean_pressure(pi)
    molar_mass = mean_molar_mass(mole_fractions, database)
    scale_height = R_GAS * temperature_k / (molar_mass * gravity_m_s2)
    zi = scale_height * np.log(surface_pressure_pa / pi)
    z = 0.5 * (zi[:-1] + zi[1:])
    rho = p * molar_mass / (R_GAS * temperature_k)
    layer_mass_pressure = pi[:-1] - pi[1:]
    reconstructed = rho * gravity_m_s2 * (zi[1:] - zi[:-1])
    residual = float(np.max(np.abs(reconstructed - layer_mass_pressure) / np.maximum(layer_mass_pressure, 1e-30)))
    return AtmosphericProfile(p, pi, z, np.full(layers, temperature_k), rho, molar_mass,
                              dict(mole_fractions), residual)


__all__ = [
    "logarithmic_cell_mean_pressure",
    "logarithmic_pressure_interfaces",
    "mean_molar_mass",
    "solve_isothermal_hydrostatic",
    "solve_temperature_profile_hydrostatic",
]
