from __future__ import annotations

"""Conservative one-dimensional eddy transport and quench diagnostics."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import block_diag, csr_matrix, diags


@dataclass(frozen=True, slots=True)
class VerticalTransportResult:
    time_s: np.ndarray
    altitude_center_m: np.ndarray
    mole_fraction: Mapping[str, np.ndarray]
    final_mole_fraction: Mapping[str, np.ndarray]
    initial_column_moles_m2: Mapping[str, float]
    final_column_moles_m2: Mapping[str, float]
    column_conservation_relative: float
    minimum_raw_mole_fraction: float
    maximum_sum_mole_fraction: float
    method: str
    nfev: int
    njev: int
    nlu: int
    message: str


@dataclass(frozen=True, slots=True)
class QuenchDiagnostic:
    mixing_timescale_s: np.ndarray
    chemical_timescale_s: np.ndarray
    quenched: np.ndarray
    crossing_indices: tuple[int, ...]


def mixing_timescale_s(mixing_length_m: np.ndarray | float,
                       eddy_diffusivity_m2_s: np.ndarray | float) -> np.ndarray:
    """Return the conventional diffusive mixing timescale L^2 / Kzz."""
    length = np.asarray(mixing_length_m, dtype=float)
    kzz = np.asarray(eddy_diffusivity_m2_s, dtype=float)
    if np.any(~np.isfinite(length)) or np.any(length < 0):
        raise ValueError("mixing length must be finite and non-negative")
    if np.any(~np.isfinite(kzz)) or np.any(kzz <= 0):
        raise ValueError("eddy diffusivity must be finite and positive")
    return np.asarray(length**2 / kzz, dtype=float)


def quench_diagnostic(*, chemical_timescale_s: np.ndarray,
                      mixing_length_m: np.ndarray | float,
                      eddy_diffusivity_m2_s: np.ndarray | float) -> QuenchDiagnostic:
    """Compare chemical and mixing timescales without prescribing a quench height.

    A level is flagged as transport-dominated when tau_chem > tau_mix. Crossing
    indices locate changes in that inequality and can be mapped to a pressure or
    altitude grid by the caller.
    """
    chemical = np.asarray(chemical_timescale_s, dtype=float)
    if chemical.ndim != 1 or chemical.size == 0:
        raise ValueError("chemical_timescale_s must be a non-empty 1-D array")
    if np.any(np.isnan(chemical)) or np.any(chemical < 0):
        raise ValueError("chemical timescales must be non-negative and not NaN")
    mixing = np.broadcast_to(
        mixing_timescale_s(mixing_length_m, eddy_diffusivity_m2_s), chemical.shape
    ).astype(float, copy=True)
    quenched = chemical > mixing
    crossings = tuple(int(index + 1) for index in np.flatnonzero(quenched[1:] != quenched[:-1]))
    return QuenchDiagnostic(mixing, chemical.copy(), quenched, crossings)


def _interface_kzz(eddy_diffusivity_m2_s: np.ndarray | float, layers: int) -> np.ndarray:
    kzz = np.asarray(eddy_diffusivity_m2_s, dtype=float)
    if kzz.ndim == 0:
        interfaces = np.full(layers + 1, float(kzz), dtype=float)
    elif kzz.shape == (layers + 1,):
        interfaces = kzz.copy()
    elif kzz.shape == (layers,):
        interfaces = np.empty(layers + 1, dtype=float)
        interfaces[0] = kzz[0]
        interfaces[-1] = kzz[-1]
        if layers > 1:
            # Harmonic averaging avoids an artificially large interface flux when
            # one neighbouring layer has a strongly limiting diffusivity.
            interfaces[1:-1] = 2.0 * kzz[:-1] * kzz[1:] / np.maximum(kzz[:-1] + kzz[1:], 1e-300)
    else:
        raise ValueError("eddy diffusivity must be scalar, per-layer, or per-interface")
    if np.any(~np.isfinite(interfaces)) or np.any(interfaces <= 0):
        raise ValueError("eddy diffusivity must be finite and positive")
    return interfaces


def eddy_diffusion_flux_mol_m2_s(*, altitude_interface_m: np.ndarray,
                                 total_molar_density_mol_m3: np.ndarray,
                                 mole_fraction: np.ndarray,
                                 eddy_diffusivity_m2_s: np.ndarray | float) -> np.ndarray:
    """Return upward-positive finite-volume eddy flux at layer interfaces.

    Boundary fluxes are zero. Interior flux uses
    ``Phi = -Kzz * n * df/dz`` with arithmetic interface molar density.
    """
    interfaces = np.asarray(altitude_interface_m, dtype=float)
    density = np.asarray(total_molar_density_mol_m3, dtype=float)
    fraction = np.asarray(mole_fraction, dtype=float)
    if interfaces.ndim != 1 or interfaces.size < 3 or np.any(~np.isfinite(interfaces)) or np.any(np.diff(interfaces) <= 0):
        raise ValueError("altitude_interface_m must be finite, increasing, and define at least two layers")
    layers = interfaces.size - 1
    if density.shape != (layers,) or fraction.shape != (layers,):
        raise ValueError("density and mole fraction must have one value per layer")
    if np.any(~np.isfinite(density)) or np.any(density <= 0):
        raise ValueError("total molar density must be finite and positive")
    if np.any(~np.isfinite(fraction)) or np.any(fraction < 0):
        raise ValueError("mole fraction must be finite and non-negative")
    kzz = _interface_kzz(eddy_diffusivity_m2_s, layers)
    centers = 0.5 * (interfaces[:-1] + interfaces[1:])
    flux = np.zeros(layers + 1, dtype=float)
    if layers > 1:
        distance = centers[1:] - centers[:-1]
        interface_density = 0.5 * (density[:-1] + density[1:])
        flux[1:-1] = -kzz[1:-1] * interface_density * np.diff(fraction) / distance
    return flux


def integrate_eddy_diffusion(*, altitude_interface_m: np.ndarray,
                             total_molar_density_mol_m3: np.ndarray,
                             initial_mole_fractions: Mapping[str, np.ndarray],
                             eddy_diffusivity_m2_s: np.ndarray | float,
                             duration_s: float,
                             method: str = "BDF",
                             relative_tolerance: float = 1e-8,
                             absolute_tolerance: float = 1e-12,
                             sample_times_s: np.ndarray | None = None) -> VerticalTransportResult:
    """Integrate conservative eddy diffusion for one or more gas tracers.

    Total molar density and Kzz are held fixed during this operator-split transport
    step. Species share the same Kzz, so a complete composition that initially sums
    to unity remains normalized apart from numerical integration error. Zero-flux
    top/bottom boundaries make each species column inventory independently closed.
    """
    interfaces = np.asarray(altitude_interface_m, dtype=float)
    density = np.asarray(total_molar_density_mol_m3, dtype=float)
    if interfaces.ndim != 1 or interfaces.size < 3 or np.any(~np.isfinite(interfaces)) or np.any(np.diff(interfaces) <= 0):
        raise ValueError("altitude_interface_m must be finite, increasing, and define at least two layers")
    layers = interfaces.size - 1
    if density.shape != (layers,) or np.any(~np.isfinite(density)) or np.any(density <= 0):
        raise ValueError("total molar density must contain one finite positive value per layer")
    if not initial_mole_fractions:
        raise ValueError("initial_mole_fractions cannot be empty")
    keys = tuple(sorted(str(key) for key in initial_mole_fractions))
    if len(keys) != len(initial_mole_fractions):
        raise ValueError("duplicate tracer keys after string normalization")
    initial = np.empty((len(keys), layers), dtype=float)
    for species_index, key in enumerate(keys):
        values = np.asarray(initial_mole_fractions[key], dtype=float)
        if values.shape != (layers,) or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"invalid mole-fraction profile for {key!r}")
        initial[species_index] = values
    initial_sum = np.sum(initial, axis=0)
    if np.any(initial_sum > 1.0 + 2e-10):
        raise ValueError("tracked mole fractions cannot sum above one in any layer")

    kzz = _interface_kzz(eddy_diffusivity_m2_s, layers)
    duration = float(duration_s)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration_s must be finite and non-negative")
    if method not in {"BDF", "Radau"}:
        raise ValueError("transport method must be BDF or Radau")
    if relative_tolerance <= 0 or absolute_tolerance <= 0:
        raise ValueError("transport tolerances must be positive")

    thickness = np.diff(interfaces)
    centers = 0.5 * (interfaces[:-1] + interfaces[1:])

    # Linear sparsity pattern: each layer couples only to itself and immediate
    # vertical neighbours for a given tracer; tracers are independent in this
    # eddy-only operator.
    single = diags(
        [np.ones(layers - 1), np.ones(layers), np.ones(layers - 1)],
        offsets=[-1, 0, 1], shape=(layers, layers), dtype=bool,
    ).tocsr()
    sparsity = block_diag([single] * len(keys), format="csr")

    def rhs(_time: float, state_flat: np.ndarray) -> np.ndarray:
        state = np.asarray(state_flat, dtype=float).reshape(len(keys), layers)
        derivative = np.empty_like(state)
        for species_index in range(len(keys)):
            # Tiny negative internal BDF iterates are clipped only while evaluating
            # the nonlinear residual; significant negative accepted states are
            # diagnosed after integration.
            fraction = np.maximum(state[species_index], 0.0)
            flux = eddy_diffusion_flux_mol_m2_s(
                altitude_interface_m=interfaces,
                total_molar_density_mol_m3=density,
                mole_fraction=fraction,
                eddy_diffusivity_m2_s=kzz,
            )
            concentration_tendency = -(flux[1:] - flux[:-1]) / thickness
            derivative[species_index] = concentration_tendency / density
        return derivative.ravel()

    y0 = initial.ravel()
    if sample_times_s is None:
        t_eval = np.asarray([0.0, duration], dtype=float) if duration > 0 else np.asarray([0.0])
    else:
        t_eval = np.asarray(sample_times_s, dtype=float)
        if t_eval.ndim != 1 or t_eval.size == 0 or np.any(~np.isfinite(t_eval)) or np.any(np.diff(t_eval) < 0):
            raise ValueError("sample_times_s must be a finite sorted non-empty 1-D array")
        if t_eval[0] < 0 or t_eval[-1] > duration:
            raise ValueError("sample_times_s must lie within [0, duration_s]")
        if t_eval[0] != 0.0:
            t_eval = np.concatenate(([0.0], t_eval))
        if duration > 0 and t_eval[-1] != duration:
            t_eval = np.concatenate((t_eval, [duration]))

    if duration == 0:
        time = np.asarray([0.0])
        raw = y0[:, None]
        nfev = njev = nlu = 0
        message = "zero-duration integration"
    else:
        solution = solve_ivp(
            rhs, (0.0, duration), y0, method=method, t_eval=t_eval,
            rtol=float(relative_tolerance), atol=float(absolute_tolerance),
            jac_sparsity=sparsity,
        )
        if not solution.success:
            raise RuntimeError(f"vertical eddy transport solver failed: {solution.message}")
        time = np.asarray(solution.t, dtype=float)
        raw = np.asarray(solution.y, dtype=float)
        nfev = int(solution.nfev)
        njev = int(getattr(solution, "njev", 0) or 0)
        nlu = int(getattr(solution, "nlu", 0) or 0)
        message = str(solution.message)

    minimum_raw = float(np.min(raw)) if raw.size else 0.0
    negative_limit = max(50.0 * float(absolute_tolerance), 1e-12)
    if minimum_raw < -negative_limit:
        raise RuntimeError(f"vertical transport produced negative mole fraction {minimum_raw:g}")
    clipped = np.maximum(raw, 0.0)
    trajectories: dict[str, np.ndarray] = {}
    final: dict[str, np.ndarray] = {}
    initial_columns: dict[str, float] = {}
    final_columns: dict[str, float] = {}
    max_relative = 0.0
    for species_index, key in enumerate(keys):
        values = clipped[species_index * layers:(species_index + 1) * layers].T
        trajectories[key] = values.copy()
        final[key] = values[-1].copy()
        initial_column = float(np.sum(density * initial[species_index] * thickness))
        final_column = float(np.sum(density * values[-1] * thickness))
        initial_columns[key] = initial_column
        final_columns[key] = final_column
        max_relative = max(
            max_relative,
            abs(final_column - initial_column) / max(abs(initial_column), 1e-30),
        )

    final_sum = np.sum(np.stack(tuple(final.values())), axis=0)
    max_sum = float(np.max(final_sum))
    normalization_tolerance = max(2e-7, 20.0 * relative_tolerance)
    if max_sum > 1.0 + normalization_tolerance:
        raise RuntimeError(f"vertical transport violated mole-fraction normalization: {max_sum:g}")
    if max_relative > max(2e-7, 20.0 * relative_tolerance):
        raise RuntimeError(f"vertical transport column conservation failed: {max_relative:g}")

    return VerticalTransportResult(
        time_s=time.copy(),
        altitude_center_m=centers.copy(),
        mole_fraction=trajectories,
        final_mole_fraction=final,
        initial_column_moles_m2=initial_columns,
        final_column_moles_m2=final_columns,
        column_conservation_relative=float(max_relative),
        minimum_raw_mole_fraction=minimum_raw,
        maximum_sum_mole_fraction=max_sum,
        method=method,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
        message=message,
    )
