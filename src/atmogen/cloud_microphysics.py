from __future__ import annotations

"""Reduced but physically defined cloud/aerosol particle microphysics primitives."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq

from .optics import mie_sphere_efficiencies


@dataclass(frozen=True, slots=True)
class ParticlePopulation:
    """Spherical lognormal particle population.

    ``geometric_std`` is the multiplicative geometric standard deviation sigma_g,
    so ln(r) has standard deviation ln(sigma_g). ``composition`` is normalized as
    a descriptive material fraction; optical/thermophysical mixture properties are
    supplied separately until sourced material data are available.
    """

    composition: Mapping[str, float]
    number_concentration_m3: float
    median_radius_m: float
    geometric_std: float
    particle_density_kg_m3: float

    def __post_init__(self) -> None:
        if not self.composition:
            raise ValueError("particle composition cannot be empty")
        composition: dict[str, float] = {}
        for raw_key, raw_value in self.composition.items():
            key = str(raw_key)
            value = float(raw_value)
            if not key or not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid particle composition {raw_key!r}: {raw_value!r}")
            if value > 0:
                composition[key] = composition.get(key, 0.0) + value
        total = sum(composition.values())
        if total <= 0:
            raise ValueError("particle composition must contain a positive fraction")
        object.__setattr__(self, "composition", {key: value / total for key, value in composition.items()})
        for name in ("number_concentration_m3", "median_radius_m", "particle_density_kg_m3"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        sigma = float(self.geometric_std)
        if not np.isfinite(sigma) or sigma < 1.0:
            raise ValueError("geometric_std must be finite and at least 1")

    @property
    def log_sigma(self) -> float:
        return float(np.log(self.geometric_std))

    def radius_moment(self, order: float) -> float:
        """Analytical expectation E[r^order] for the lognormal distribution."""
        k = float(order)
        if not np.isfinite(k):
            raise ValueError("moment order must be finite")
        return float(self.median_radius_m**k * np.exp(0.5 * k * k * self.log_sigma**2))

    @property
    def effective_radius_m(self) -> float:
        return self.radius_moment(3.0) / self.radius_moment(2.0)

    @property
    def mass_concentration_kg_m3(self) -> float:
        mean_particle_volume = 4.0 * np.pi / 3.0 * self.radius_moment(3.0)
        return float(self.number_concentration_m3 * self.particle_density_kg_m3 * mean_particle_volume)


@dataclass(frozen=True, slots=True)
class SettlingResult:
    terminal_velocity_m_s: float
    reynolds_number: float
    drag_coefficient: float
    cunningham_factor: float
    regime: str


@dataclass(frozen=True, slots=True)
class SedimentationResult:
    mass_concentration_kg_m3: float
    mass_flux_kg_m2_s: float
    mass_weighted_velocity_m_s: float
    residence_time_s: float | None
    maximum_reynolds_number: float
    quadrature_order: int


@dataclass(frozen=True, slots=True)
class ParticleOpticalCoefficients:
    wavelength_m: np.ndarray
    extinction_m_inv: np.ndarray
    scattering_m_inv: np.ndarray
    absorption_m_inv: np.ndarray
    single_scattering_albedo: np.ndarray
    asymmetry_g: np.ndarray
    quadrature_order: int


@dataclass(frozen=True, slots=True)
class PrecipitationStepResult:
    remaining_condensate_kg_m2: np.ndarray
    downward_transfer_kg_m2: np.ndarray
    reevaporated_kg_m2: np.ndarray
    surface_precipitation_kg_m2: float
    mass_closure_relative: float


def cunningham_slip_correction(particle_radius_m: float, mean_free_path_m: float = 0.0) -> float:
    """Davies/Cunningham slip correction in terms of sphere radius."""
    radius = float(particle_radius_m)
    mean_free_path = float(mean_free_path_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("particle_radius_m must be finite and positive")
    if not np.isfinite(mean_free_path) or mean_free_path < 0:
        raise ValueError("mean_free_path_m must be finite and non-negative")
    if mean_free_path == 0:
        return 1.0
    return float(
        1.0 + mean_free_path / radius
        * (1.257 + 0.4 * np.exp(-1.1 * radius / mean_free_path))
    )


def sphere_drag_coefficient(reynolds_number: float) -> float:
    """Schiller-Naumann sphere drag correlation with high-Re constant branch."""
    reynolds = float(reynolds_number)
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("reynolds_number must be finite and positive")
    if reynolds < 1000.0:
        return float(24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687))
    return 0.44


def terminal_settling_velocity(*, particle_radius_m: float, particle_density_kg_m3: float,
                               gas_density_kg_m3: float, gas_dynamic_viscosity_pa_s: float,
                               gravity_m_s2: float, mean_free_path_m: float = 0.0) -> SettlingResult:
    """Terminal downward speed for a spherical particle.

    The Stokes-Cunningham expression is accepted only when its resulting Reynolds
    number is below 0.1. Otherwise the velocity is solved from buoyancy/gravity and
    quadratic drag with the Schiller-Naumann sphere correlation. Slip correction is
    not extrapolated outside the creeping-flow branch.
    """
    radius = float(particle_radius_m)
    rho_p = float(particle_density_kg_m3)
    rho_g = float(gas_density_kg_m3)
    viscosity = float(gas_dynamic_viscosity_pa_s)
    gravity = float(gravity_m_s2)
    for name, value in (("particle_radius_m", radius), ("particle_density_kg_m3", rho_p),
                        ("gas_density_kg_m3", rho_g), ("gas_dynamic_viscosity_pa_s", viscosity),
                        ("gravity_m_s2", gravity)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if rho_p <= rho_g:
        raise ValueError("particle density must exceed gas density for downward settling")
    slip = cunningham_slip_correction(radius, mean_free_path_m)
    delta_rho = rho_p - rho_g
    stokes = 2.0 / 9.0 * delta_rho * gravity * radius**2 / viscosity * slip
    re_stokes = rho_g * (2.0 * radius) * stokes / viscosity
    if re_stokes < 0.1:
        cd = 24.0 / max(re_stokes, 1e-300)
        return SettlingResult(float(stokes), float(re_stokes), float(cd), slip, "Stokes-Cunningham")

    force = 4.0 / 3.0 * np.pi * radius**3 * delta_rho * gravity
    area = np.pi * radius**2

    def residual(velocity: float) -> float:
        if velocity <= 0:
            return -force
        reynolds = rho_g * (2.0 * radius) * velocity / viscosity
        cd = sphere_drag_coefficient(reynolds)
        drag = 0.5 * cd * rho_g * area * velocity**2
        return drag - force

    high = max(stokes, np.sqrt(2.0 * force / (0.44 * rho_g * area)), 1e-9)
    for _ in range(80):
        if residual(high) > 0:
            break
        high *= 2.0
    else:  # pragma: no cover - pathological physical input
        raise RuntimeError("could not bracket particle terminal velocity")
    velocity = float(brentq(residual, 0.0, high, xtol=1e-14, rtol=1e-12, maxiter=200))
    reynolds = rho_g * (2.0 * radius) * velocity / viscosity
    cd = sphere_drag_coefficient(reynolds)
    return SettlingResult(velocity, float(reynolds), cd, 1.0, "Schiller-Naumann")


def _radius_quadrature(population: ParticlePopulation, order: int) -> tuple[np.ndarray, np.ndarray]:
    points = int(order)
    if points < 1 or points > 128:
        raise ValueError("quadrature order must be in [1, 128]")
    if population.geometric_std == 1.0 or points == 1:
        return np.asarray([population.median_radius_m]), np.asarray([1.0])
    nodes, weights = hermgauss(points)
    radii = population.median_radius_m * np.exp(np.sqrt(2.0) * population.log_sigma * nodes)
    probabilities = weights / np.sqrt(np.pi)
    return np.asarray(radii, dtype=float), np.asarray(probabilities, dtype=float)


def sedimentation_mass_flux(*, population: ParticlePopulation, gas_density_kg_m3: float,
                            gas_dynamic_viscosity_pa_s: float, gravity_m_s2: float,
                            mean_free_path_m: float = 0.0, layer_depth_m: float | None = None,
                            quadrature_order: int = 24) -> SedimentationResult:
    """Integrate particle mass times settling speed over a lognormal distribution."""
    radii, weights = _radius_quadrature(population, quadrature_order)
    particle_masses = 4.0 / 3.0 * np.pi * radii**3 * population.particle_density_kg_m3
    velocities = np.empty_like(radii)
    reynolds = np.empty_like(radii)
    for index, radius in enumerate(radii):
        settled = terminal_settling_velocity(
            particle_radius_m=float(radius),
            particle_density_kg_m3=population.particle_density_kg_m3,
            gas_density_kg_m3=gas_density_kg_m3,
            gas_dynamic_viscosity_pa_s=gas_dynamic_viscosity_pa_s,
            gravity_m_s2=gravity_m_s2,
            mean_free_path_m=mean_free_path_m,
        )
        velocities[index] = settled.terminal_velocity_m_s
        reynolds[index] = settled.reynolds_number
    mean_mass = float(np.sum(weights * particle_masses))
    mean_mass_velocity = float(np.sum(weights * particle_masses * velocities))
    mass_concentration = population.number_concentration_m3 * mean_mass
    mass_flux = population.number_concentration_m3 * mean_mass_velocity
    mass_weighted_velocity = mean_mass_velocity / max(mean_mass, 1e-300)
    residence = None
    if layer_depth_m is not None:
        depth = float(layer_depth_m)
        if not np.isfinite(depth) or depth < 0:
            raise ValueError("layer_depth_m must be finite and non-negative")
        residence = depth / max(mass_weighted_velocity, 1e-300)
    return SedimentationResult(
        mass_concentration_kg_m3=float(mass_concentration),
        mass_flux_kg_m2_s=float(mass_flux),
        mass_weighted_velocity_m_s=float(mass_weighted_velocity),
        residence_time_s=None if residence is None else float(residence),
        maximum_reynolds_number=float(np.max(reynolds)),
        quadrature_order=len(radii),
    )


def particle_optical_coefficients(*, population: ParticlePopulation, wavelength_m: np.ndarray,
                                  particle_refractive_index: np.ndarray | complex,
                                  medium_refractive_index: float = 1.0,
                                  quadrature_order: int = 20) -> ParticleOpticalCoefficients:
    """Integrate Mie extinction/scattering over a lognormal radius distribution."""
    wavelength = np.asarray(wavelength_m, dtype=float)
    if wavelength.ndim != 1 or wavelength.size == 0 or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelength_m must be a finite positive 1-D array")
    indices = np.asarray(particle_refractive_index, dtype=np.complex128)
    if indices.ndim == 0:
        indices = np.full(wavelength.shape, complex(indices), dtype=np.complex128)
    if indices.shape != wavelength.shape:
        raise ValueError("particle_refractive_index must be scalar or match wavelength_m")
    if np.any(~np.isfinite(indices.real)) or np.any(~np.isfinite(indices.imag)) or np.any(indices.real <= 0) or np.any(indices.imag < 0):
        raise ValueError("particle indices must use n + i*kappa with n > 0 and kappa >= 0")

    radii, weights = _radius_quadrature(population, quadrature_order)
    beta_ext = np.zeros_like(wavelength)
    beta_sca = np.zeros_like(wavelength)
    beta_abs = np.zeros_like(wavelength)
    asymmetry = np.zeros_like(wavelength)
    for wave_index, wave in enumerate(wavelength):
        ext_cross = 0.0
        sca_cross = 0.0
        abs_cross = 0.0
        g_sca_cross = 0.0
        for radius, weight in zip(radii, weights, strict=True):
            efficiency = mie_sphere_efficiencies(
                radius_m=float(radius), wavelength_m=float(wave),
                particle_refractive_index=complex(indices[wave_index]),
                medium_refractive_index=medium_refractive_index,
            )
            geometric = np.pi * radius**2
            ext = geometric * efficiency.q_ext
            sca = geometric * efficiency.q_sca
            abs_ = geometric * efficiency.q_abs
            ext_cross += float(weight * ext)
            sca_cross += float(weight * sca)
            abs_cross += float(weight * abs_)
            g_sca_cross += float(weight * sca * efficiency.asymmetry_g)
        beta_ext[wave_index] = population.number_concentration_m3 * ext_cross
        beta_sca[wave_index] = population.number_concentration_m3 * sca_cross
        beta_abs[wave_index] = population.number_concentration_m3 * abs_cross
        asymmetry[wave_index] = g_sca_cross / max(sca_cross, 1e-300)
    omega = np.divide(beta_sca, beta_ext, out=np.zeros_like(beta_sca), where=beta_ext > 0)
    return ParticleOpticalCoefficients(
        wavelength_m=wavelength.copy(),
        extinction_m_inv=beta_ext,
        scattering_m_inv=beta_sca,
        absorption_m_inv=beta_abs,
        single_scattering_albedo=np.clip(omega, 0.0, 1.0),
        asymmetry_g=np.clip(asymmetry, -1.0, 1.0),
        quadrature_order=len(radii),
    )


def precipitation_step(*, condensate_kg_m2: np.ndarray, layer_thickness_m: np.ndarray,
                       settling_velocity_m_s: np.ndarray, timestep_s: float,
                       evaporation_timescale_s: np.ndarray | None = None) -> PrecipitationStepResult:
    """Route one operator-split settling step downward with optional re-evaporation.

    Layer index 0 is the surface layer and increasing indices are higher altitude.
    Material leaving layer ``j`` may enter ``j-1`` or reach the surface for ``j=0``.
    Only mass present at the beginning of the step may cross an interface, preventing
    a large timestep from teleporting newly arrived condensate through many layers.
    The optional evaporation timescale belongs to the receiving layer and controls
    the survival of falling material during its one-layer traversal.
    """
    mass = np.asarray(condensate_kg_m2, dtype=float)
    thickness = np.asarray(layer_thickness_m, dtype=float)
    velocity = np.asarray(settling_velocity_m_s, dtype=float)
    if mass.ndim != 1 or mass.size == 0 or thickness.shape != mass.shape or velocity.shape != mass.shape:
        raise ValueError("condensate, thickness, and settling velocity must be same-shape non-empty 1-D arrays")
    if np.any(~np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("condensate mass must be finite and non-negative")
    if np.any(~np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("layer thickness must be finite and positive")
    if np.any(~np.isfinite(velocity)) or np.any(velocity < 0):
        raise ValueError("settling velocity must be finite and non-negative")
    dt = float(timestep_s)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("timestep_s must be finite and non-negative")
    if evaporation_timescale_s is None:
        evaporation = np.full_like(mass, np.inf)
    else:
        evaporation = np.asarray(evaporation_timescale_s, dtype=float)
        if evaporation.shape != mass.shape or np.any(np.isnan(evaporation)) or np.any(evaporation <= 0):
            raise ValueError("evaporation timescale must be positive, non-NaN, and match the layer shape")

    fraction_leaving = 1.0 - np.exp(-velocity * dt / thickness)
    transfer_raw = mass * np.clip(fraction_leaving, 0.0, 1.0)
    remaining = mass - transfer_raw
    transfer_surviving = np.zeros_like(mass)
    reevaporated = np.zeros_like(mass)
    surface = 0.0
    for source in range(mass.size):
        falling = float(transfer_raw[source])
        if falling <= 0:
            continue
        if source == 0:
            # No lower atmospheric layer is represented; material crossing the
            # lowest interface is counted as surface precipitation.
            surface += falling
            transfer_surviving[source] = falling
            continue
        receiver = source - 1
        travel_time = thickness[receiver] / max(velocity[source], 1e-300)
        if np.isinf(evaporation[receiver]):
            survival = 1.0
        else:
            survival = float(np.exp(-travel_time / evaporation[receiver]))
        surviving = falling * survival
        evaporated = falling - surviving
        remaining[receiver] += surviving
        transfer_surviving[source] = surviving
        reevaporated[receiver] += evaporated

    initial_total = float(np.sum(mass))
    final_total = float(np.sum(remaining) + np.sum(reevaporated) + surface)
    closure = abs(final_total - initial_total) / max(initial_total, 1.0)
    if closure > 2e-12:
        raise RuntimeError(f"precipitation routing failed mass closure: {closure:g}")
    return PrecipitationStepResult(
        remaining_condensate_kg_m2=remaining,
        downward_transfer_kg_m2=transfer_surviving,
        reevaporated_kg_m2=reevaporated,
        surface_precipitation_kg_m2=float(surface),
        mass_closure_relative=float(closure),
    )
