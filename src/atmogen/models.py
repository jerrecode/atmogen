from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


class Fidelity(str, Enum):
    FAST = "FAST"
    STANDARD = "STANDARD"
    HIGH = "HIGH"
    REFERENCE = "REFERENCE"


@dataclass(frozen=True, slots=True)
class PlanetPhysicalState:
    """Bulk boundary conditions, in SI units."""

    radius_m: float
    gravity_m_s2: float
    surface_pressure_pa: float
    initial_surface_temperature_k: float = 288.15
    surface_albedo_initial: float = 0.15
    internal_heat_flux_w_m2: float = 0.0

    def __post_init__(self) -> None:
        for name in ("radius_m", "gravity_m_s2", "surface_pressure_pa", "initial_surface_temperature_k"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.surface_albedo_initial < 1:
            raise ValueError("surface_albedo_initial must be in [0, 1)")
        if self.internal_heat_flux_w_m2 < 0:
            raise ValueError("internal_heat_flux_w_m2 cannot be negative")


@dataclass(frozen=True, slots=True)
class StellarSpectrum:
    """Top-of-atmosphere spectral irradiance. Wavelength is in metres."""

    wavelength_m: NDArray[np.float64]
    flux_w_m2_m: NDArray[np.float64]
    provenance: str

    def __post_init__(self) -> None:
        wave = np.asarray(self.wavelength_m, dtype=float)
        flux = np.asarray(self.flux_w_m2_m, dtype=float)
        if wave.ndim != 1 or wave.size < 2 or flux.shape != wave.shape:
            raise ValueError("stellar spectrum arrays must be same-length 1-D arrays")
        if np.any(~np.isfinite(wave)) or np.any(np.diff(wave) <= 0) or np.any(wave <= 0):
            raise ValueError("wavelength_m must be finite, positive, and increasing")
        if np.any(~np.isfinite(flux)) or np.any(flux < 0):
            raise ValueError("flux_w_m2_m must be finite and non-negative")

    @property
    def bolometric_flux_w_m2(self) -> float:
        return float(np.trapezoid(self.flux_w_m2_m, self.wavelength_m))


@dataclass(frozen=True, slots=True)
class ElementInventory:
    """Element amounts plus an optional molecular initial-state hint, in mol."""

    element_moles: Mapping[str, float]
    initial_species_moles: Mapping[str, float] = field(default_factory=dict)
    semantics: str = "element_inventory"

    def __post_init__(self) -> None:
        if not self.element_moles:
            raise ValueError("element_moles cannot be empty")
        for group_name, values in (("element_moles", self.element_moles), ("initial_species_moles", self.initial_species_moles)):
            for key, value in values.items():
                if not key or not np.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {group_name} entry {key!r}: {value!r}")


@dataclass(frozen=True, slots=True)
class SurfaceReservoirs:
    """Surface-accessible species masses in kg."""

    species_mass_kg: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.species_mass_kg.items():
            if not key or not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid surface reservoir {key!r}: {value!r}")


@dataclass(frozen=True, slots=True)
class SolverSettings:
    fidelity: Fidelity = Fidelity.FAST
    vertical_layers: int | None = None
    top_pressure_pa: float = 0.1
    chemistry_mode: str = "equilibrium"  # equilibrium | fixed_species
    radiation_mode: str = "semi_gray_spectral_shortwave"
    cloud_mode: str = "equilibrium_bulk"  # equilibrium_bulk | lognormal_sedimentation
    activity_model: str = "auto"  # auto | ideal | nrtl
    liquid_phase_split: bool = True
    vertical_transport_mode: str = "none"  # none | eddy_diffusion
    eddy_diffusivity_m2_s: float = 50.0
    cloud_suspended_fraction: float = 0.01
    cloud_condensate_column_cap_kg_m2: float = 0.2
    cloud_particle_median_radius_m: float = 10e-6
    cloud_particle_geometric_std: float = 1.4
    cloud_particle_density_kg_m3: float | None = None
    cloud_refractive_index_real: float | None = None
    cloud_refractive_index_imag: float = 0.0
    gas_dynamic_viscosity_pa_s: float = 1.8e-5
    cloud_microphysics_timestep_s: float = 3600.0
    cloud_reevaporation_timescale_s: float | None = None
    cloud_quadrature_order: int = 12
    max_iterations: int = 40
    relative_temperature_tolerance: float = 2e-5
    composition_tolerance: float = 1e-9
    energy_tolerance_w_m2: float = 0.05
    relaxation: float = 0.55
    allow_fidelity_fallback: bool = True

    @property
    def resolved_layers(self) -> int:
        if self.vertical_layers is not None:
            return self.vertical_layers
        return {Fidelity.FAST: 24, Fidelity.STANDARD: 48, Fidelity.HIGH: 96, Fidelity.REFERENCE: 160}[self.fidelity]

    def __post_init__(self) -> None:
        if self.resolved_layers < 4:
            raise ValueError("vertical_layers must be at least 4")
        if self.top_pressure_pa <= 0:
            raise ValueError("top_pressure_pa must be positive")
        if self.chemistry_mode not in {"equilibrium", "fixed_species"}:
            raise ValueError("chemistry_mode must be equilibrium or fixed_species")
        if self.cloud_mode not in {"equilibrium_bulk", "lognormal_sedimentation"}:
            raise ValueError("cloud_mode must be equilibrium_bulk or lognormal_sedimentation")
        if self.activity_model not in {"auto", "ideal", "nrtl"}:
            raise ValueError("activity_model must be auto, ideal, or nrtl")
        if not isinstance(self.liquid_phase_split, bool):
            raise TypeError("liquid_phase_split must be bool")
        if self.vertical_transport_mode not in {"none", "eddy_diffusion"}:
            raise ValueError("vertical_transport_mode must be none or eddy_diffusion")
        positive_names = (
            "eddy_diffusivity_m2_s", "cloud_condensate_column_cap_kg_m2",
            "cloud_particle_median_radius_m", "cloud_particle_geometric_std",
            "gas_dynamic_viscosity_pa_s", "cloud_microphysics_timestep_s",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.cloud_suspended_fraction) or not 0 <= self.cloud_suspended_fraction <= 1:
            raise ValueError("cloud_suspended_fraction must be in [0, 1]")
        if self.cloud_particle_geometric_std < 1:
            raise ValueError("cloud_particle_geometric_std must be at least 1")
        if self.cloud_particle_density_kg_m3 is not None and (
            not np.isfinite(self.cloud_particle_density_kg_m3) or self.cloud_particle_density_kg_m3 <= 0
        ):
            raise ValueError("cloud_particle_density_kg_m3 must be positive when supplied")
        if self.cloud_refractive_index_real is not None and (
            not np.isfinite(self.cloud_refractive_index_real) or self.cloud_refractive_index_real <= 0
        ):
            raise ValueError("cloud_refractive_index_real must be positive when supplied")
        if not np.isfinite(self.cloud_refractive_index_imag) or self.cloud_refractive_index_imag < 0:
            raise ValueError("cloud_refractive_index_imag must be finite and non-negative")
        if self.cloud_reevaporation_timescale_s is not None and (
            not np.isfinite(self.cloud_reevaporation_timescale_s) or self.cloud_reevaporation_timescale_s <= 0
        ):
            raise ValueError("cloud_reevaporation_timescale_s must be positive when supplied")
        if not isinstance(self.cloud_quadrature_order, int) or not 1 <= self.cloud_quadrature_order <= 128:
            raise ValueError("cloud_quadrature_order must be an integer in [1, 128]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class AtmosphericProfile:
    pressure_pa: NDArray[np.float64]
    pressure_interface_pa: NDArray[np.float64]
    altitude_m: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    density_kg_m3: NDArray[np.float64]
    mean_molar_mass_kg_mol: float
    mole_fractions: Mapping[str, float]
    hydrostatic_relative_residual: float


@dataclass(frozen=True, slots=True)
class LiquidPhaseState:
    """One thermodynamic liquid phase, with global phase inventory."""

    phase_fraction_mol: float
    species_moles: Mapping[str, float]
    mole_fractions: Mapping[str, float]
    species_mass_kg: Mapping[str, float]
    density_kg_m3: float | None
    volume_m3: float | None
    activity_coefficients: Mapping[str, float]
    activity_model: str


@dataclass(frozen=True, slots=True)
class PhaseReservoirResult:
    atmospheric_mass_kg: Mapping[str, float]
    liquid_mass_kg: Mapping[str, float]
    solid_mass_kg: Mapping[str, float]
    liquid_volume_m3: Mapping[str, float]
    latent_heat_flux_w_m2: float
    mass_closure_relative: float
    fallbacks: tuple[str, ...]
    surface_vapor_mole_fractions: Mapping[str, float] = field(default_factory=dict)
    liquid_phases: tuple[LiquidPhaseState, ...] = ()
    activity_model: str = "ideal"


@dataclass(frozen=True, slots=True)
class CloudResult:
    condensate_mass_kg: Mapping[str, float]
    effective_radius_m: float
    optical_depth_visible: float
    model: str


@dataclass(frozen=True, slots=True)
class VerticalProcessResult:
    layer_thickness_m: NDArray[np.float64]
    eddy_diffusivity_m2_s: NDArray[np.float64]
    mixing_timescale_s: NDArray[np.float64]
    cloud_condensate_kg_m2: NDArray[np.float64]
    cloud_mass_concentration_kg_m3: NDArray[np.float64]
    cloud_number_concentration_m3: NDArray[np.float64]
    cloud_settling_velocity_m_s: NDArray[np.float64]
    cloud_sedimentation_flux_kg_m2_s: NDArray[np.float64]
    optical_wavelength_m: NDArray[np.float64]
    cloud_extinction_m_inv: NDArray[np.float64]
    cloud_scattering_m_inv: NDArray[np.float64]
    cloud_absorption_m_inv: NDArray[np.float64]
    cloud_single_scattering_albedo: NDArray[np.float64]
    cloud_asymmetry_g: NDArray[np.float64]
    precipitation_downward_kg_m2: NDArray[np.float64]
    precipitation_reevaporated_kg_m2: NDArray[np.float64]
    surface_precipitation_kg_m2: float
    reevaporation_latent_cooling_w_m2: float
    mass_closure_relative: float
    model: str
    fallbacks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpectralResult:
    shortwave_wavelength_m: NDArray[np.float64]
    incident_shortwave_w_m2_m: NDArray[np.float64]
    reflected_shortwave_w_m2_m: NDArray[np.float64]
    transmitted_shortwave_fraction: NDArray[np.float64]
    rayleigh_optical_depth: NDArray[np.float64]
    thermal_wavelength_m: NDArray[np.float64]
    outgoing_thermal_w_m2_m: NDArray[np.float64]
    spectral_albedo: NDArray[np.float64]
    bond_albedo: float
    geometric_albedo_approx: float
    visible_srgb: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class EnergyBudget:
    incoming_global_mean_w_m2: float
    absorbed_shortwave_w_m2: float
    outgoing_longwave_w_m2: float
    internal_heat_w_m2: float
    imbalance_w_m2: float
    longwave_optical_depth: float


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    converged: bool
    iterations: int
    temperature_relative_residual: float
    composition_absolute_residual: float
    energy_imbalance_w_m2: float
    residual_history: tuple[Mapping[str, float], ...]


@dataclass(frozen=True, slots=True)
class PlanetChemistryResult:
    atmosphere: AtmosphericProfile
    surface: PhaseReservoirResult
    clouds: CloudResult
    vertical: VerticalProcessResult
    spectra: SpectralResult
    energy_budget: EnergyBudget
    convergence: ConvergenceReport
    diagnostics: Mapping[str, object]
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ColumnInput:
    planet: PlanetPhysicalState
    inventory: ElementInventory
    surface: SurfaceReservoirs = field(default_factory=SurfaceReservoirs)
    stellar_flux_scale: float = 1.0

    def __post_init__(self) -> None:
        scale = float(self.stellar_flux_scale)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("stellar_flux_scale must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ColumnBatchInput:
    columns: tuple[ColumnInput, ...]
    star: StellarSpectrum
