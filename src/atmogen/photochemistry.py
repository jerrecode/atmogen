from __future__ import annotations

"""Wavelength-resolved photolysis primitives.

Photochemical rates use photon spectral flux in photons m^-2 s^-1 m^-1 and
cross sections in m^2. The first-order photolysis coefficient is

    J = integral sigma(lambda, T) * q(lambda) * Phi(lambda) dlambda

with units s^-1. This module deliberately separates radiative attenuation from
reaction kinetics: callers may supply top-of-atmosphere or layer-local actinic
fluxes from progressively more capable radiative-transfer backends.
"""

from dataclasses import dataclass

import numpy as np

from .database import ProvenanceClass

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0


@dataclass(frozen=True, slots=True)
class PhotolysisData:
    """Spectral cross-section and quantum-yield data for one photolysis reaction."""

    reaction_key: str
    parent_species: str
    wavelength_m: np.ndarray
    cross_section_m2: np.ndarray
    quantum_yield: np.ndarray
    provenance_class: ProvenanceClass
    source: str
    validity: str
    temperature_k: float | None = None

    def __post_init__(self) -> None:
        if not self.reaction_key or not self.parent_species:
            raise ValueError("photolysis data require reaction_key and parent_species")
        wave = np.asarray(self.wavelength_m, dtype=float)
        sigma = np.asarray(self.cross_section_m2, dtype=float)
        yield_ = np.asarray(self.quantum_yield, dtype=float)
        if wave.ndim != 1 or wave.size < 2:
            raise ValueError("photolysis wavelength grid must be a 1-D array with at least two points")
        if sigma.shape != wave.shape or yield_.shape != wave.shape:
            raise ValueError("photolysis spectral arrays must share the wavelength shape")
        if np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError("photolysis wavelengths must be finite, positive, and strictly increasing")
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("photolysis cross sections must be finite and non-negative")
        if np.any(~np.isfinite(yield_)) or np.any(yield_ < 0):
            raise ValueError("photolysis quantum yields must be finite and non-negative")
        if self.temperature_k is not None and (
            not np.isfinite(self.temperature_k) or self.temperature_k <= 0
        ):
            raise ValueError("photolysis temperature_k must be finite and positive when supplied")
        object.__setattr__(self, "wavelength_m", wave.copy())
        object.__setattr__(self, "cross_section_m2", sigma.copy())
        object.__setattr__(self, "quantum_yield", yield_.copy())


def spectral_irradiance_to_photon_flux(
    wavelength_m: np.ndarray,
    spectral_irradiance_w_m2_m: np.ndarray,
) -> np.ndarray:
    """Convert spectral energy irradiance to photon spectral flux."""
    wave = np.asarray(wavelength_m, dtype=float)
    irradiance = np.asarray(spectral_irradiance_w_m2_m, dtype=float)
    if wave.shape != irradiance.shape or wave.ndim != 1:
        raise ValueError("wavelength and irradiance must be same-shape 1-D arrays")
    if np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError("wavelength_m must be finite, positive, and increasing")
    if np.any(~np.isfinite(irradiance)) or np.any(irradiance < 0):
        raise ValueError("spectral irradiance must be finite and non-negative")
    return irradiance * wave / (H_PLANCK * C_LIGHT)


def attenuate_actinic_photon_flux(
    top_photon_flux_m2_s_m: np.ndarray,
    optical_depth: np.ndarray,
) -> np.ndarray:
    """Beer-Lambert attenuation for one spectrum or a stack of layer optical depths."""
    flux = np.asarray(top_photon_flux_m2_s_m, dtype=float)
    tau = np.asarray(optical_depth, dtype=float)
    if flux.ndim != 1:
        raise ValueError("top photon flux must be a 1-D spectral array")
    if np.any(~np.isfinite(flux)) or np.any(flux < 0):
        raise ValueError("top photon flux must be finite and non-negative")
    if tau.shape[-1:] != flux.shape:
        raise ValueError("optical_depth final dimension must match photon-flux wavelength dimension")
    if np.any(~np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError("optical depth must be finite and non-negative")
    return flux * np.exp(-np.clip(tau, 0.0, 745.0))


def photolysis_rate_s1(
    data: PhotolysisData,
    photon_flux_m2_s_m: np.ndarray,
    wavelength_m: np.ndarray | None = None,
) -> float:
    """Integrate one local photolysis coefficient J [s^-1]."""
    flux = np.asarray(photon_flux_m2_s_m, dtype=float)
    if wavelength_m is None:
        if flux.shape != data.wavelength_m.shape:
            raise ValueError("photon flux must share the photolysis wavelength grid when wavelength_m is omitted")
        local_flux = flux
    else:
        wave = np.asarray(wavelength_m, dtype=float)
        if wave.ndim != 1 or flux.shape != wave.shape:
            raise ValueError("supplied wavelength and photon-flux arrays must be same-shape 1-D arrays")
        if np.any(~np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
            raise ValueError("supplied wavelength grid must be finite, positive, and increasing")
        if np.any(~np.isfinite(flux)) or np.any(flux < 0):
            raise ValueError("photon flux must be finite and non-negative")
        local_flux = np.interp(data.wavelength_m, wave, flux, left=0.0, right=0.0)
    if np.any(~np.isfinite(local_flux)) or np.any(local_flux < 0):
        raise ValueError("photon flux must be finite and non-negative")
    integrand = data.cross_section_m2 * data.quantum_yield * local_flux
    value = float(np.trapezoid(integrand, data.wavelength_m))
    if not np.isfinite(value) or value < 0:
        raise FloatingPointError(f"invalid photolysis coefficient for {data.reaction_key!r}: {value!r}")
    return value


def column_photolysis_rates_s1(
    data: PhotolysisData,
    top_photon_flux_m2_s_m: np.ndarray,
    cumulative_optical_depth: np.ndarray,
    wavelength_m: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate J for each vertical layer from cumulative top-down optical depth.

    ``cumulative_optical_depth`` may be ``(layers, wavelengths)`` or a single
    wavelength vector. The result is always a 1-D layer vector.
    """
    top_flux = np.asarray(top_photon_flux_m2_s_m, dtype=float)
    if wavelength_m is None:
        wave = data.wavelength_m
        if top_flux.shape != wave.shape:
            raise ValueError("top photon flux must share the photolysis wavelength grid")
    else:
        wave = np.asarray(wavelength_m, dtype=float)
        if wave.ndim != 1 or top_flux.shape != wave.shape:
            raise ValueError("top photon flux and wavelength_m must be same-shape 1-D arrays")
    tau = np.asarray(cumulative_optical_depth, dtype=float)
    if tau.ndim == 1:
        tau = tau[None, :]
    if tau.ndim != 2 or tau.shape[1] != top_flux.size:
        raise ValueError("cumulative_optical_depth must have shape (layers, wavelengths)")
    local = attenuate_actinic_photon_flux(top_flux, tau)
    return np.asarray([
        photolysis_rate_s1(data, spectrum, wavelength_m=wave)
        for spectrum in local
    ], dtype=float)
