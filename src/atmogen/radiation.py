from __future__ import annotations

from typing import Mapping

import numpy as np

from .database import BUILTIN_DATABASE, ChemicalDatabase
from .models import StellarSpectrum

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_BOLTZMANN = 1.380649e-23
SIGMA_SB = 5.670374419e-8
N_AVOGADRO = 6.02214076e23


def planck_radiance_w_m2_sr_m(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    wave = np.asarray(wavelength_m, dtype=np.float64)
    if temperature_k <= 0 or np.any(wave <= 0):
        raise ValueError("temperature and wavelength must be positive")
    x = np.clip(H_PLANCK * C_LIGHT / (wave * K_BOLTZMANN * temperature_k), 1e-12, 700.0)
    return (2.0 * H_PLANCK * C_LIGHT**2 / wave**5) / np.expm1(x)


def blackbody_stellar_spectrum(temperature_k: float, bolometric_flux_w_m2: float,
                               wavelength_m: np.ndarray | None = None) -> StellarSpectrum:
    wave = np.geomspace(1e-7, 1e-4, 1024) if wavelength_m is None else np.asarray(wavelength_m, float)
    shape = np.pi * planck_radiance_w_m2_sr_m(wave, temperature_k)
    shape *= bolometric_flux_w_m2 / np.trapezoid(shape, wave)
    return StellarSpectrum(wave, shape, f"blackbody fallback at {temperature_k:g} K, normalized to supplied bolometric flux")


def beer_lambert_transmission(optical_depth: np.ndarray | float) -> np.ndarray:
    tau = np.asarray(optical_depth, dtype=float)
    if np.any(tau < 0):
        raise ValueError("optical depth cannot be negative")
    return np.exp(-np.clip(tau, 0.0, 745.0))


def rayleigh_cross_section_m2(species: str, wavelength_m: np.ndarray,
                              database: ChemicalDatabase = BUILTIN_DATABASE) -> np.ndarray:
    reference = database.get(species).rayleigh_cross_section_550_m2
    if reference is None:
        return np.zeros_like(np.asarray(wavelength_m, float))
    return reference * (550e-9 / np.asarray(wavelength_m, float)) ** 4


def rayleigh_optical_depth(*, wavelength_m: np.ndarray, surface_pressure_pa: float,
                           gravity_m_s2: float, mole_fractions: Mapping[str, float],
                           database: ChemicalDatabase = BUILTIN_DATABASE) -> np.ndarray:
    mean_mm = sum(database.get(k).molar_mass_kg_mol * float(x) for k, x in mole_fractions.items())
    total_molecules_m2 = surface_pressure_pa / gravity_m_s2 / mean_mm * N_AVOGADRO
    tau = np.zeros_like(np.asarray(wavelength_m, float))
    for key, fraction in mole_fractions.items():
        tau += total_molecules_m2 * float(fraction) * rayleigh_cross_section_m2(key, wavelength_m, database)
    return tau


def fresnel_normal_reflectance(n1: complex, n2: complex) -> float:
    if n1 + n2 == 0:
        raise ValueError("n1 + n2 cannot be zero")
    return float(abs((n1 - n2) / (n1 + n2)) ** 2)


def _cie_1931_fit(wavelength_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic fit to the CIE 1931 2-degree CMFs (Wyman et al. 2013)."""
    w = np.asarray(wavelength_nm, float)
    def g(mu: float, left: float, right: float) -> np.ndarray:
        scale = np.where(w < mu, left, right)
        return np.exp(-0.5 * ((w - mu) * scale) ** 2)
    x = 1.056 * g(599.8, 0.0264, 0.0323) + 0.362 * g(442.0, 0.0624, 0.0374) - 0.065 * g(501.1, 0.0490, 0.0382)
    y = 0.821 * g(568.8, 0.0213, 0.0247) + 0.286 * g(530.9, 0.0613, 0.0322)
    z = 1.217 * g(437.0, 0.0845, 0.0278) + 0.681 * g(459.0, 0.0385, 0.0725)
    return np.maximum(x, 0), np.maximum(y, 0), np.maximum(z, 0)


def spectrum_to_srgb(wavelength_m: np.ndarray, radiance: np.ndarray) -> tuple[float, float, float]:
    wave_nm = np.asarray(wavelength_m, float) * 1e9
    values = np.maximum(np.asarray(radiance, float), 0)
    mask = (wave_nm >= 360) & (wave_nm <= 830)
    if np.count_nonzero(mask) < 2 or not np.any(values[mask] > 0):
        return (0.0, 0.0, 0.0)
    xbar, ybar, zbar = _cie_1931_fit(wave_nm[mask])
    xyz = np.asarray([np.trapezoid(values[mask] * q, wave_nm[mask]) for q in (xbar, ybar, zbar)])
    xyz /= max(float(xyz[1]), 1e-30)
    rgb = np.asarray([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]) @ xyz
    rgb = np.maximum(rgb, 0)
    rgb /= max(float(np.max(rgb)), 1e-30)
    encoded = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return tuple(float(v) for v in np.clip(encoded, 0, 1))


def longwave_optical_depth(*, surface_pressure_pa: float, gravity_m_s2: float,
                           mole_fractions: Mapping[str, float],
                           additional_species_column_kg_m2: Mapping[str, float] | None = None,
                           database: ChemicalDatabase = BUILTIN_DATABASE) -> float:
    mean_mm = sum(database.get(k).molar_mass_kg_mol * float(x) for k, x in mole_fractions.items())
    column_mass = surface_pressure_pa / gravity_m_s2
    # FAST mode uses a bounded square-root column proxy. It represents saturation
    # and pressure-broadened band overlap better than linear gray absorption while
    # remaining explicitly empirical. Coefficients therefore have implied units
    # (kg m^-2)^-1/2 and are never presented as laboratory cross sections.
    tau = 0.0
    for key, x in mole_fractions.items():
        mass_fraction = float(x) * database.get(key).molar_mass_kg_mol / mean_mm
        species_column = max(column_mass * mass_fraction, 0.0)
        tau += database.get(key).longwave_column_coefficient_fast * np.sqrt(species_column)
    for key, species_column in (additional_species_column_kg_m2 or {}).items():
        raw = database.get(key).longwave_column_coefficient_fast * np.sqrt(max(float(species_column), 0.0))
        # A surface-supplied trace vapor cannot open unlimited independent gray
        # bands. Smooth saturation prevents the FAST proxy from inventing a moist
        # runaway solely because the same broad bands are counted repeatedly.
        tau += 1.5 * raw / (1.5 + raw)
    return float(max(tau, 0.0))
