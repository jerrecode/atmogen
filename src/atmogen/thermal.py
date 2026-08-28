from __future__ import annotations

"""Reduced-order vertical temperature-profile solvers.

The radiative-convective backends intentionally separate three fidelity levels:
isothermal compatibility, dry gray static-stability adjustment, and a bounded
single-condensable dilute saturated adjustment. The saturated formulation is an
approximate pseudoadiabatic/saturation-adiabatic lapse-rate constraint, not a full
reversible moist-convection or layerwise radiative-equilibrium calculation.
"""

from typing import Mapping

import numpy as np

from .chemistry import R_GAS
from .database import BUILTIN_DATABASE, ChemicalDatabase, canonical_species


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


def _dry_carrier_properties(
    mole_fractions: Mapping[str, float],
    *,
    condensible: str,
    database: ChemicalDatabase,
) -> tuple[float, float, float]:
    """Return carrier molar mass, R_specific, and Cp_specific excluding condensable."""
    vapor = canonical_species(condensible)
    carrier = {
        canonical_species(key): max(float(value), 0.0)
        for key, value in mole_fractions.items()
        if canonical_species(key) != vapor and float(value) > 0
    }
    total = float(sum(carrier.values()))
    if total <= 0:
        raise ValueError(
            "dilute saturated convection requires a non-condensable carrier atmosphere"
        )
    normalized = {key: value / total for key, value in carrier.items()}
    molar_mass = float(
        sum(
            fraction * database.get(key).molar_mass_kg_mol
            for key, fraction in normalized.items()
        )
    )
    cp_molar = float(
        sum(
            fraction * database.get(key).heat_capacity_j_mol_k
            for key, fraction in normalized.items()
        )
    )
    if molar_mass <= 0 or cp_molar <= R_GAS:
        raise ValueError("carrier atmosphere has invalid thermodynamic properties")
    r_specific = R_GAS / molar_mass
    cp_specific = cp_molar / molar_mass
    return molar_mass, r_specific, cp_specific


def dilute_saturated_log_pressure_gradient(
    *,
    pressure_pa: float,
    temperature_k: float,
    mole_fractions: Mapping[str, float],
    condensible: str,
    max_saturation_mixing_ratio: float = 0.25,
    allow_estimated_saturation: bool = False,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[float | None, float | None, str | None]:
    """Approximate saturated ∇ = dlnT/dlnP for one dilute condensable.

    Uses the standard approximate saturated/pseudoadiabatic lapse-rate expression

        Gamma_m = g/c_pd * (1 + L r_s/(R_d T))
                         / (1 + epsilon L^2 r_s/(c_pd R_d T^2))

    and hydrostatic ideal-gas conversion ∇_m = Gamma_m R_d/g. ``r_s`` is the
    saturation vapor/dry-carrier mass mixing ratio. The approximation is rejected
    instead of clipped when the condensable becomes non-dilute, saturation data are
    estimated without explicit permission, or phase-specific latent data are
    insufficient.
    """
    p = float(pressure_pa)
    t = float(temperature_k)
    if not np.isfinite(p) or p <= 0 or not np.isfinite(t) or t <= 0:
        raise ValueError("pressure_pa and temperature_k must be finite and positive")
    limit = float(max_saturation_mixing_ratio)
    if not np.isfinite(limit) or not 0 < limit <= 1:
        raise ValueError("max_saturation_mixing_ratio must be in (0, 1]")

    key = canonical_species(condensible)
    species = database.get(key)
    latent = species.latent_heat_j_kg
    if latent is None or not np.isfinite(latent) or latent <= 0:
        return None, None, f"{key}: latent heat unavailable; dry convective constraint used"
    if species.freezing_point_k is not None and t < species.freezing_point_k:
        return None, None, (
            f"{key}: saturated profile entered the solid regime but only one generic "
            "latent-heat datum is bundled; dry constraint used instead of inventing "
            "a sublimation latent heat"
        )

    # Lazy import keeps the saturation/phase module independent of thermal.py.
    from .phase import saturation_pressure_pa

    saturation_pressure, saturation_note = saturation_pressure_pa(key, t)
    if saturation_pressure is None:
        return None, None, saturation_note or (
            f"{key}: saturation pressure unavailable; dry convective constraint used"
        )
    if (
        saturation_note
        and "estimated" in saturation_note.lower()
        and not allow_estimated_saturation
    ):
        return None, None, (
            f"{key}: only an estimated saturation-pressure relation is bundled; "
            "set moist_allow_estimated_saturation=true to use it"
        )
    e_s = float(saturation_pressure)
    if not np.isfinite(e_s) or e_s < 0:
        raise RuntimeError(f"{key}: saturation-pressure backend returned invalid data")
    if e_s >= p:
        return None, None, (
            f"{key}: saturation vapor pressure reaches/exceeds total pressure; "
            "dilute saturated approximation is invalid"
        )

    carrier_molar_mass, r_d, cp_d = _dry_carrier_properties(
        mole_fractions, condensible=key, database=database
    )
    epsilon = species.molar_mass_kg_mol / carrier_molar_mass
    r_s = epsilon * e_s / max(p - e_s, 1e-300)
    if not np.isfinite(r_s) or r_s < 0:
        raise RuntimeError(f"{key}: computed saturation mixing ratio is invalid")
    if r_s > limit:
        return None, float(r_s), (
            f"{key}: saturation mixing ratio {r_s:.6g} kg/kg exceeds configured "
            f"dilute limit {limit:.6g}; dry convective constraint used"
        )

    numerator = 1.0 + latent * r_s / (r_d * t)
    denominator = 1.0 + (
        epsilon * latent * latent * r_s / (cp_d * r_d * t * t)
    )
    dry_gradient = r_d / cp_d
    moist_gradient = dry_gradient * numerator / denominator
    if not np.isfinite(moist_gradient) or moist_gradient <= 0:
        raise RuntimeError(f"{key}: saturated lapse-rate calculation is invalid")
    # In the dilute saturated approximation latent heating cannot make the lapse
    # steeper than the corresponding dry carrier adiabat.
    moist_gradient = min(float(moist_gradient), float(dry_gradient))
    return moist_gradient, float(r_s), saturation_note


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
    anchor = tau[0] + 2.0 / 3.0
    ratio = (tau + 2.0 / 3.0) / max(float(anchor), 1e-300)
    profile = surface_temperature_k * np.power(ratio, 0.25)
    profile[0] = surface_temperature_k
    return np.asarray(profile, dtype=np.float64)


def _validate_adjustment_arrays(
    pressure_pa: np.ndarray, temperature_k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    return pressure, original


def dry_convective_adjustment(
    *,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    mole_fractions: Mapping[str, float],
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove superadiabatic instability while leaving stable layers unchanged."""
    pressure, original = _validate_adjustment_arrays(pressure_pa, temperature_k)
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


def dilute_saturated_convective_adjustment(
    *,
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    mole_fractions: Mapping[str, float],
    condensible: str,
    max_saturation_mixing_ratio: float = 0.25,
    allow_estimated_saturation: bool = False,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Apply a single-condensable dilute saturated stability constraint.

    Layers for which the saturated approximation is outside its data or dilute
    validity domain use the dry-mixture adiabatic constraint instead. The returned
    masks distinguish layers actually temperature-adjusted from layers where the
    saturated constraint itself was valid and used.
    """
    pressure, original = _validate_adjustment_arrays(pressure_pa, temperature_k)
    adjusted = original.copy()
    changed = np.zeros(adjusted.size, dtype=bool)
    saturated_used = np.zeros(adjusted.size, dtype=bool)
    dry_gradient = dry_adiabatic_log_pressure_gradient(mole_fractions, database)
    notes: list[str] = []

    for upper in range(1, adjusted.size):
        gradient, _mixing_ratio, note = dilute_saturated_log_pressure_gradient(
            pressure_pa=float(pressure[upper - 1]),
            temperature_k=float(adjusted[upper - 1]),
            mole_fractions=mole_fractions,
            condensible=condensible,
            max_saturation_mixing_ratio=max_saturation_mixing_ratio,
            allow_estimated_saturation=allow_estimated_saturation,
            database=database,
        )
        if note and note not in notes:
            notes.append(note)
        if gradient is None:
            gradient = dry_gradient
        else:
            saturated_used[upper] = True
        minimum_stable_upper = adjusted[upper - 1] * (
            pressure[upper] / pressure[upper - 1]
        ) ** float(gradient)
        if adjusted[upper] < minimum_stable_upper:
            adjusted[upper] = minimum_stable_upper
            changed[upper] = True

    return adjusted, changed, saturated_used, tuple(notes)


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


def solve_dilute_saturated_radiative_convective_profile(
    *,
    pressure_pa: np.ndarray,
    surface_pressure_pa: float,
    surface_temperature_k: float,
    longwave_optical_depth_surface: float,
    mole_fractions: Mapping[str, float],
    condensible: str,
    max_saturation_mixing_ratio: float = 0.25,
    allow_estimated_saturation: bool = False,
    optical_depth_pressure_exponent: float = DEFAULT_GRAY_OPTICAL_DEPTH_PRESSURE_EXPONENT,
    database: ChemicalDatabase = BUILTIN_DATABASE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build a gray radiative profile and impose bounded dilute saturated stability."""
    radiative = gray_radiative_temperature_profile(
        pressure_pa=pressure_pa,
        surface_pressure_pa=surface_pressure_pa,
        surface_temperature_k=surface_temperature_k,
        longwave_optical_depth_surface=longwave_optical_depth_surface,
        optical_depth_pressure_exponent=optical_depth_pressure_exponent,
    )
    return dilute_saturated_convective_adjustment(
        pressure_pa=pressure_pa,
        temperature_k=radiative,
        mole_fractions=mole_fractions,
        condensible=condensible,
        max_saturation_mixing_ratio=max_saturation_mixing_ratio,
        allow_estimated_saturation=allow_estimated_saturation,
        database=database,
    )


__all__ = [
    "DEFAULT_GRAY_OPTICAL_DEPTH_PRESSURE_EXPONENT",
    "dilute_saturated_convective_adjustment",
    "dilute_saturated_log_pressure_gradient",
    "dry_adiabatic_log_pressure_gradient",
    "dry_convective_adjustment",
    "gray_radiative_temperature_profile",
    "mixture_molar_heat_capacity_j_mol_k",
    "solve_dilute_saturated_radiative_convective_profile",
    "solve_dry_radiative_convective_profile",
]
