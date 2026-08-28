from __future__ import annotations

"""Reduced-order vertical temperature-profile solvers.

The current radiative-convective backend is deliberately a dry gray model. It is a
physically constrained replacement for an isothermal profile at STANDARD+ fidelity,
not a claim of moist-convective or nongray radiative equilibrium. Condensing/moist
adiabats require species-specific latent thermodynamics and are kept separate.
"""

from typing import Mapping

import numpy as np

from .chemistry import R_GAS
from .database import BUILTIN_DATABASE, ChemicalDatabase


DEFAULT_GRAY_OPTICAL_DEPTH_PRESSURE_EXPONENT = 2.0


def mixture_molar_heat_capacity_j_mol_k(
    mole_fractions: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> float:
    """Return ideal-mixture molar constant-pressure heat capacity."""
    positive = {str(k): max(float(v), 0.0) for k, v in mole_fractions.items()}
    total = float(sum(positive.values()))
    if total <= 0:
        raise ValueError("mole_fractions must contain positive material")
    cp = sum(
        (value / total) * database.get(key).heat_capacity_j_mol_k
        for key, value in positive.items()
    )
    if not np.isfinite(cp) or cp <= R_GAS:
        raise ValueError("mixture heat capacity must be finite and greater than R")
    return float(cp)


def dry_adiabatic_log_pressure_gradient(
    mole_fractions: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> float:
    """Ideal-gas dry adiabatic gradient ∇_ad = R/Cp in dlnT/dlnP coordinates."""
    return float(
        R_GAS / mixture_molar_heat_capacity_j_mol_k(mole_fractions, database)
    )


def gray_radiative_temperature_profile(
    *,
    pressure_pa: np.ndarray,
    surface_pressure_pa: float,
    surface_temperature_k: float,
    longwave_optical_depth_surface: float,
    optical_depth_pressure_exponent: float = DEFAULT_GRAY_OPTICAL_DEPTH_PRESSURE_EXPONENT,
) -> np.ndarray:
    """Return a gray Eddington-shaped layer temperature profile.

    The optical-depth law is

        τ(P) = τ_s (P/P_s)^n,

    and the radiative shape follows T^4 ∝ (τ + 2/3). AtmosphericProfile stores
    layer-center rather than interface temperatures, while downstream code has long
    treated temperature_k[0] as the surface-temperature proxy. Therefore the shape
    is normalized at the deepest supplied layer center so its first value is exactly
    ``surface_temperature_k``. The true lower interface remains represented by
    ``surface_pressure_pa`` for the optical-depth scaling.
    """
    pressure = np.asarray(pressure_pa, dtype=float)
    if (
        pressure.ndim != 1
        or pressure.size < 1
        or np.any(~np.isfinite(pressure))
        or np.any(pressure <= 0)
    ):
        raise ValueError("pressure_pa must be a finite positive 1-D array")
    if pressure.size > 1 and np.any(np.diff(pressure) >= 0):
        raise ValueError("pressure_pa must decrease from surface toward the top")
    if not np.isfinite(surface_pressure_pa) or surface_pressure_pa <= 0:
        raise ValueError("surface_pressure_pa must be finite and positive")
    if np.any(pressure > surface_pressure_pa * (1.0 + 1e-12)):
        raise ValueError("layer pressure cannot exceed surface pressure")
    if not np.isfinite(surface_temperature_k) or surface_temperature_k <= 0:
        raise ValueError("surface_temperature_k must be finite and positive")
    tau_s = float(longwave_optical_depth_surface)
    exponent = float(optical_depth_pressure_exponent)
    if not np.isfinite(tau_s) or tau_s < 0:
        raise ValueError(
            "longwave_optical_depth_surface must be finite and non-negative"
        )
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("optical_depth_pressure_exponent must be finite and positive")

    tau = tau_s * np.power(
        np.clip(pressure / surface_pressure_pa, 0.0, 1.0), exponent
    )
    # Anchor at the deepest represented layer to preserve the established public
    # temperature_k[0] surface-proxy contract.
    anchor = tau[0] + 2.0 / 3.0
    ratio = (tau + 2.0 / 3.0) / max(float(anchor), 1e-300)
    profile = surface_temperature_k * np.power(ratio, 0.25)
    profile[0] = surface_temperature_k
    return np.asarray(profile, dtype=np.float64)


def dry_convective_adjustment(
    *,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    mole_fractions: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove superadiabatic instability while leaving stable layers unchanged.

    Pressure must run from high to low (surface toward top). For every adjacent pair,
    the upper temperature is raised if the implied dlnT/dlnP exceeds the ideal dry
    adiabatic gradient. The returned mask identifies adjusted layers. This is a dry
    static-stability correction only; latent heating is intentionally not included.
    """
    pressure = np.asarray(pressure_pa, dtype=float)
    original = np.asarray(temperature_k, dtype=float)
    if pressure.shape != original.shape or pressure.ndim != 1 or pressure.size < 1:
        raise ValueError("pressure and temperature must be same-length 1-D arrays")
    if (
        np.any(~np.isfinite(pressure))
        or np.any(pressure <= 0)
        or (pressure.size > 1 and np.any(np.diff(pressure) >= 0))
    ):
        raise ValueError("pressure must be finite, positive, and strictly decreasing")
    if np.any(~np.isfinite(original)) or np.any(original <= 0):
        raise ValueError("temperature must be finite and positive")

    adjusted = original.copy()
    changed = np.zeros(adjusted.size, dtype=bool)
    nabla_ad = dry_adiabatic_log_pressure_gradient(mole_fractions, database)
    for upper in range(1, adjusted.size):
        minimum_stable_upper = adjusted[upper - 1] * (
            pressure[upper] / pressure[upper - 1]
        ) ** nabla_ad
        if adjusted[upper] < minimum_stable_upper:
            adjusted[upper] = minimum_stable_upper
            changed[upper] = True
    return adjusted, changed


def solve_dry_radiative_convective_profile(
    *,
    pressure_pa: np.ndarray,
    surface_pressure_pa: float,
    surface_temperature_k: float,
    longwave_optical_depth_surface: float,
    mole_fractions: Mapping[str, float],
    optical_depth_pressure_exponent: float = DEFAULT_GRAY_OPTICAL_DEPTH_PRESSURE_EXPONENT,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a gray radiative profile and impose dry static stability."""
    radiative = gray_radiative_temperature_profile(
        pressure_pa=pressure_pa,
        surface_pressure_pa=surface_pressure_pa,
        surface_temperature_k=surface_temperature_k,
        longwave_optical_depth_surface=longwave_optical_depth_surface,
        optical_depth_pressure_exponent=optical_depth_pressure_exponent,
    )
    return dry_convective_adjustment(
        pressure_pa=pressure_pa,
        temperature_k=radiative,
        mole_fractions=mole_fractions,
        database=database,
    )


__all__ = [
    "DEFAULT_GRAY_OPTICAL_DEPTH_PRESSURE_EXPONENT",
    "dry_adiabatic_log_pressure_gradient",
    "dry_convective_adjustment",
    "gray_radiative_temperature_profile",
    "mixture_molar_heat_capacity_j_mol_k",
    "solve_dry_radiative_convective_profile",
]
