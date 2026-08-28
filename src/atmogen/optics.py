from __future__ import annotations

"""Wavelength-resolved optical primitives for planetary materials and particles.

The Mie backend follows the homogeneous-sphere Lorenz-Mie series in the
Bohren-Huffman/Wiscombe family of algorithms. The logarithmic derivative of the
Riccati-Bessel function is evaluated by downward recurrence, avoiding the fragile
direct complex-Bessel evaluation used by many compact teaching implementations.

atmogen consistently represents absorbing material indices as ``n + i*kappa`` with
``kappa >= 0``. Literature and legacy Mie codes often use ``n - i*kappa`` because of
a different harmonic time convention; benchmark inputs must therefore be converted
rather than copied blindly.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class MieEfficiencies:
    """Dimensionless homogeneous-sphere scattering efficiencies."""

    q_sca: float
    q_abs: float
    q_ext: float
    q_back: float
    asymmetry_g: float
    single_scattering_albedo: float
    size_parameter: float
    series_terms: int


@dataclass(frozen=True, slots=True)
class FresnelReflectance:
    """Power reflectance for s, p, and unpolarized incident light."""

    s: float
    p: float
    unpolarized: float
    transmitted_angle_rad: complex


def absorption_coefficient_m_inv(refractive_index: complex, wavelength_m: float) -> float:
    """Return Beer-Lambert absorption coefficient ``alpha = 4*pi*kappa/lambda``."""
    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength_m must be finite and positive")
    index = complex(refractive_index)
    if index.real <= 0 or index.imag < 0:
        raise ValueError("refractive index must use n + i*kappa with n > 0 and kappa >= 0")
    return float(4.0 * np.pi * index.imag / wavelength)


def lorentz_lorenz_mix(
    refractive_indices: Mapping[str, complex],
    volume_fractions: Mapping[str, float],
) -> complex:
    """Effective complex refractive index from the Lorentz-Lorenz mixing rule.

    The relation is appropriate as a molecular-scale isotropic effective-medium
    approximation. It is not used silently for emulsions, aerosols, or separated
    macroscopic phases, where morphology-specific rules are required.
    """
    weighted: dict[str, float] = {}
    for raw_key, raw_fraction in volume_fractions.items():
        key = str(raw_key)
        fraction = float(raw_fraction)
        if not np.isfinite(fraction) or fraction < 0:
            raise ValueError("volume fractions must be finite and non-negative")
        if fraction <= 0:
            continue
        if key not in refractive_indices:
            raise KeyError(f"missing refractive index for {key!r}")
        weighted[key] = weighted.get(key, 0.0) + fraction
    total = sum(weighted.values())
    if total <= 0:
        raise ValueError("volume_fractions must contain a positive amount")

    lorentz = 0.0j
    for key, fraction in weighted.items():
        index = complex(refractive_indices[key])
        if index.real <= 0 or index.imag < 0:
            raise ValueError("indices must use n + i*kappa with n > 0 and kappa >= 0")
        epsilon = index * index
        denominator = epsilon + 2.0
        if abs(denominator) <= 1.0e-15:
            raise FloatingPointError("Lorentz-Lorenz component denominator is singular")
        lorentz += (fraction / total) * (epsilon - 1.0) / denominator

    if abs(1.0 - lorentz) <= 1.0e-14:
        raise FloatingPointError("Lorentz-Lorenz mixture approached singular effective permittivity")
    epsilon_eff = (1.0 + 2.0 * lorentz) / (1.0 - lorentz)
    index_eff = complex(np.sqrt(epsilon_eff))
    # Choose the passive-material square-root branch.
    if index_eff.real < 0:
        index_eff = -index_eff
    if index_eff.imag < 0:
        index_eff = complex(index_eff.real, -index_eff.imag)
    return index_eff


def fresnel_reflectance(
    n_incident: float,
    n_transmitted: complex,
    incidence_angle_rad: float = 0.0,
) -> FresnelReflectance:
    """Fresnel reflectance from a non-absorbing incident medium.

    The transmitted medium may be absorbing. Restricting the incident medium to a
    real index avoids the subtle energy-flux definition required when the incident
    medium itself is absorbing. This is the relevant atmosphere-to-liquid/solid
    interface case for the current planetary renderer.
    """
    n1 = float(n_incident)
    n2 = complex(n_transmitted)
    if not np.isfinite(n1) or n1 <= 0:
        raise ValueError("n_incident must be finite, real, and positive")
    if n2.real <= 0 or n2.imag < 0 or not np.isfinite(n2.real) or not np.isfinite(n2.imag):
        raise ValueError("n_transmitted must use n + i*kappa with n > 0 and kappa >= 0")
    theta_i = float(incidence_angle_rad)
    if not np.isfinite(theta_i) or theta_i < 0 or theta_i >= 0.5 * np.pi:
        raise ValueError("incidence_angle_rad must be in [0, pi/2)")

    sin_i = np.sin(theta_i)
    cos_i = np.cos(theta_i)
    sin_t = (n1 / n2) * sin_i
    cos_t = complex(np.sqrt(1.0 - sin_t * sin_t))
    # Passive branch: the transmitted wave must not grow away from the interface.
    if cos_t.real < 0 or (abs(cos_t.real) < 1.0e-15 and cos_t.imag < 0):
        cos_t = -cos_t

    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    rs = float(np.clip(abs(r_s) ** 2, 0.0, 1.0))
    rp = float(np.clip(abs(r_p) ** 2, 0.0, 1.0))
    return FresnelReflectance(
        s=rs,
        p=rp,
        unpolarized=0.5 * (rs + rp),
        transmitted_angle_rad=complex(np.arcsin(sin_t)),
    )


def rayleigh_sphere_efficiencies(
    *,
    radius_m: float,
    wavelength_m: float,
    particle_refractive_index: complex,
    medium_refractive_index: float = 1.0,
) -> MieEfficiencies:
    """Small-particle homogeneous-sphere limit for ``x << 1``."""
    radius = float(radius_m)
    wavelength = float(wavelength_m)
    medium = float(medium_refractive_index)
    index = complex(particle_refractive_index)
    if not all(np.isfinite(v) and v > 0 for v in (radius, wavelength, medium)):
        raise ValueError("radius, wavelength, and medium refractive index must be finite and positive")
    if index.real <= 0 or index.imag < 0 or not np.isfinite(index.real) or not np.isfinite(index.imag):
        raise ValueError("particle index must use n + i*kappa with n > 0 and kappa >= 0")

    x = 2.0 * np.pi * medium * radius / wavelength
    relative = index / medium
    polarizability = (relative * relative - 1.0) / (relative * relative + 2.0)
    q_sca = float((8.0 / 3.0) * x**4 * abs(polarizability) ** 2)
    q_abs = float(max(4.0 * x * polarizability.imag, 0.0))
    q_ext = q_sca + q_abs
    omega = q_sca / q_ext if q_ext > 0 else 0.0
    # In the Rayleigh limit p(theta) is symmetric forward/backward and g -> 0.
    return MieEfficiencies(
        q_sca=q_sca,
        q_abs=q_abs,
        q_ext=q_ext,
        q_back=1.5 * q_sca,
        asymmetry_g=0.0,
        single_scattering_albedo=float(np.clip(omega, 0.0, 1.0)),
        size_parameter=float(x),
        series_terms=0,
    )


def _log_derivative_downward(z: complex, order: int) -> np.ndarray:
    """Riccati-Bessel logarithmic derivative by stable downward recurrence."""
    if z == 0:
        raise ValueError("complex size parameter cannot be zero")
    start = max(order + 25, int(np.ceil(abs(z))) + 25)
    derivative = np.zeros(start + 1, dtype=np.complex128)
    for n in range(start, 0, -1):
        nz = n / z
        denominator = derivative[n] + nz
        if abs(denominator) <= 1.0e-300:
            denominator = complex(1.0e-300, 0.0)
        derivative[n - 1] = nz - 1.0 / denominator
    return derivative[: order + 1]


def mie_sphere_efficiencies(
    *,
    radius_m: float,
    wavelength_m: float,
    particle_refractive_index: complex,
    medium_refractive_index: float = 1.0,
    max_terms: int = 20000,
) -> MieEfficiencies:
    """Lorenz-Mie efficiencies for a homogeneous spherical particle.

    The series coefficients use a Wiscombe-style logarithmic-derivative downward
    recurrence. The surrounding medium is currently required to be non-absorbing.
    For extremely large size parameters callers should eventually use a dedicated
    geometric-optics backend rather than forcing an arbitrarily long Mie series.
    """
    radius = float(radius_m)
    wavelength = float(wavelength_m)
    medium = float(medium_refractive_index)
    index = complex(particle_refractive_index)
    if not all(np.isfinite(v) and v > 0 for v in (radius, wavelength, medium)):
        raise ValueError("radius, wavelength, and medium refractive index must be finite and positive")
    if index.real <= 0 or index.imag < 0 or not np.isfinite(index.real) or not np.isfinite(index.imag):
        raise ValueError("particle index must use n + i*kappa with n > 0 and kappa >= 0")

    x = float(2.0 * np.pi * medium * radius / wavelength)
    if x < 1.0e-4:
        return rayleigh_sphere_efficiencies(
            radius_m=radius,
            wavelength_m=wavelength,
            particle_refractive_index=index,
            medium_refractive_index=medium,
        )

    n_stop = max(1, int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2.0)))
    if n_stop > int(max_terms):
        raise ValueError(
            f"Mie series requires {n_stop} terms, exceeding max_terms={max_terms}; "
            "use a large-particle/geometric-optics backend"
        )

    relative = index / medium
    derivative = _log_derivative_downward(relative * x, n_stop)
    orders = np.arange(1, n_stop + 1, dtype=np.int64)
    a_n = np.empty(n_stop, dtype=np.complex128)
    b_n = np.empty(n_stop, dtype=np.complex128)

    psi_nm1 = float(np.sin(x))
    psi_n = float(np.sin(x) / x - np.cos(x))
    # xi_n = psi_n + i*chi_n for the n+i*kappa passive convention used here.
    xi_nm1 = complex(np.sin(x), -np.cos(x))
    xi_n = complex(psi_n, -np.cos(x) / x - np.sin(x))

    for offset, n in enumerate(range(1, n_stop + 1)):
        d_n = derivative[n]
        a_term = d_n / relative + n / x
        b_term = relative * d_n + n / x
        denominator_a = a_term * xi_n - xi_nm1
        denominator_b = b_term * xi_n - xi_nm1
        if abs(denominator_a) <= 1.0e-300 or abs(denominator_b) <= 1.0e-300:
            raise FloatingPointError("singular Mie coefficient denominator")
        a_n[offset] = (a_term * psi_n - psi_nm1) / denominator_a
        b_n[offset] = (b_term * psi_n - psi_nm1) / denominator_b

        psi_np1 = (2.0 * n + 1.0) / x * psi_n - psi_nm1
        xi_np1 = (2.0 * n + 1.0) / x * xi_n - xi_nm1
        psi_nm1, psi_n = psi_n, psi_np1
        xi_nm1, xi_n = xi_n, xi_np1

    factors = 2.0 * orders + 1.0
    q_ext = float(2.0 / x**2 * np.sum(factors * np.real(a_n + b_n)))
    q_sca = float(2.0 / x**2 * np.sum(factors * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2)))
    q_abs = q_ext - q_sca
    numerical_scale = max(abs(q_ext), abs(q_sca), 1.0)
    if q_abs < 0 and abs(q_abs) <= 2.0e-11 * numerical_scale:
        q_abs = 0.0
        q_ext = q_sca
    if q_ext < -1.0e-10 or q_sca < -1.0e-10 or q_abs < -1.0e-10:
        raise FloatingPointError(
            f"unphysical Mie efficiencies q_ext={q_ext:g}, q_sca={q_sca:g}, q_abs={q_abs:g}"
        )
    q_ext = max(q_ext, 0.0)
    q_sca = max(q_sca, 0.0)
    q_abs = max(q_abs, 0.0)

    if q_sca > 0:
        adjacent = 0.0
        if n_stop > 1:
            n_adj = orders[:-1].astype(float)
            adjacent = float(np.sum(
                n_adj * (n_adj + 2.0) / (n_adj + 1.0)
                * np.real(
                    a_n[:-1] * np.conj(a_n[1:])
                    + b_n[:-1] * np.conj(b_n[1:])
                )
            ))
        cross = float(np.sum(
            factors / (orders * (orders + 1.0))
            * np.real(a_n * np.conj(b_n))
        ))
        asymmetry = float(4.0 / (x**2 * q_sca) * (adjacent + cross))
        asymmetry = float(np.clip(asymmetry, -1.0, 1.0))
    else:
        asymmetry = 0.0

    alternating = (-1.0) ** orders
    q_back = float(abs(np.sum(factors * alternating * (a_n - b_n))) ** 2 / x**2)
    omega = q_sca / q_ext if q_ext > 0 else 0.0
    return MieEfficiencies(
        q_sca=float(q_sca),
        q_abs=float(q_abs),
        q_ext=float(q_ext),
        q_back=max(q_back, 0.0),
        asymmetry_g=asymmetry,
        single_scattering_albedo=float(np.clip(omega, 0.0, 1.0)),
        size_parameter=x,
        series_terms=n_stop,
    )
