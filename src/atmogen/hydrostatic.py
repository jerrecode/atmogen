from __future__ import annotations

from typing import Mapping

import numpy as np

from .chemistry import R_GAS
from .database import BUILTIN_DATABASE, ChemicalDatabase
from .models import AtmosphericProfile


def logarithmic_pressure_interfaces(surface_pressure_pa: float, top_pressure_pa: float, layers: int) -> np.ndarray:
    surface = float(surface_pressure_pa)
    top = float(top_pressure_pa)
    if not np.isfinite(surface) or not np.isfinite(top):
        raise ValueError("surface_pressure_pa and top_pressure_pa must be finite")
    if not surface > top > 0.0:
        raise ValueError("require surface_pressure_pa > top_pressure_pa > 0")
    if not isinstance(layers, (int, np.integer)) or isinstance(layers, (bool, np.bool_)):
        raise TypeError("layers must be an integer")
    if int(layers) < 1:
        raise ValueError("layers must be positive")
    return np.geomspace(surface, top, int(layers) + 1, dtype=np.float64)


def logarithmic_cell_mean_pressure(pressure_interface_pa: np.ndarray) -> np.ndarray:
    interfaces = np.asarray(pressure_interface_pa, dtype=float)
    if interfaces.ndim != 1 or interfaces.size < 2 or np.any(~np.isfinite(interfaces)):
        raise ValueError("pressure interfaces must be a finite 1-D array")
    if np.any(interfaces <= 0) or np.any(np.diff(interfaces) >= 0):
        raise ValueError("pressure interfaces must be positive and strictly decreasing")
    return (interfaces[:-1] - interfaces[1:]) / np.log(interfaces[:-1] / interfaces[1:])


def mean_molar_mass(mole_fractions: Mapping[str, float], database: ChemicalDatabase = BUILTIN_DATABASE) -> float:
    weighted = 0.0
    total = 0.0
    for key, raw_value in mole_fractions.items():
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("mole fractions must be finite and non-negative")
        species = database.get(key)
        total += value
        weighted += value * species.molar_mass_kg_mol
    if total <= 0.0:
        raise ValueError("mole fractions must have positive total")
    return float(weighted / total)


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
    temperature = float(temperature_k)
    gravity = float(gravity_m_s2)
    if not np.isfinite(temperature) or not np.isfinite(gravity):
        raise ValueError("temperature and gravity must be finite and positive")
    if temperature <= 0.0 or gravity <= 0.0:
        raise ValueError("temperature and gravity must be finite and positive")
    pi = logarithmic_pressure_interfaces(surface_pressure_pa, top_pressure_pa, layers)
    p = logarithmic_cell_mean_pressure(pi)
    molar_mass = mean_molar_mass(mole_fractions, database)
    scale_height = R_GAS * temperature / (molar_mass * gravity)
    zi = scale_height * np.log(surface_pressure_pa / pi)
    z = 0.5 * (zi[:-1] + zi[1:])
    rho = p * molar_mass / (R_GAS * temperature)
    layer_mass_pressure = pi[:-1] - pi[1:]
    reconstructed = rho * gravity * (zi[1:] - zi[:-1])
    residual = float(np.max(np.abs(reconstructed - layer_mass_pressure) / np.maximum(layer_mass_pressure, 1e-30)))
    return AtmosphericProfile(p, pi, z, np.full(int(layers), temperature), rho, molar_mass,
                              dict(mole_fractions), residual)


__all__ = [
    "logarithmic_cell_mean_pressure",
    "logarithmic_pressure_interfaces",
    "mean_molar_mass",
    "solve_isothermal_hydrostatic",
    "solve_temperature_profile_hydrostatic",
]
