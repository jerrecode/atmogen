from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import minimize

from .database import BUILTIN_DATABASE, ChemicalDatabase, NRTLInteraction

R_GAS = 8.31446261815324
_EPS = 1.0e-14


def _normalise(mole_fractions: Mapping[str, float]) -> tuple[tuple[str, ...], np.ndarray]:
    keys = tuple(str(key) for key in mole_fractions)
    if not keys:
        raise ValueError("liquid composition cannot be empty")
    values = np.asarray([float(mole_fractions[key]) for key in keys], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("liquid mole fractions must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("liquid composition must contain a positive amount")
    return keys, values / total


class ActivityModel(Protocol):
    """Liquid activity-coefficient backend."""

    name: str

    def activity_coefficients(self, *, temperature_k: float,
                              mole_fractions: Mapping[str, float]) -> Mapping[str, float]: ...

    def reduced_mixing_gibbs(self, *, temperature_k: float,
                             mole_fractions: Mapping[str, float]) -> float: ...


class IdealActivityModel:
    """Ideal liquid solution: a_i = x_i and gamma_i = 1."""

    name = "ideal"

    def activity_coefficients(self, *, temperature_k: float,
                              mole_fractions: Mapping[str, float]) -> Mapping[str, float]:
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        keys, _ = _normalise(mole_fractions)
        return {key: 1.0 for key in keys}

    def reduced_mixing_gibbs(self, *, temperature_k: float,
                             mole_fractions: Mapping[str, float]) -> float:
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        _, x = _normalise(mole_fractions)
        return float(np.sum(x * np.log(np.maximum(x, 1.0e-300))))


class NRTLActivityModel:
    """Non-random two-liquid (NRTL) activity model.

    The directed interaction data use an isothermal energy form
    tau_ij = delta_g_ij / (R T). This is deliberately narrower than the many
    polynomial temperature correlations used in process-simulation databases;
    callers must provide parameters with explicit provenance and validity.
    """

    name = "NRTL"

    def __init__(self, interactions: Sequence[NRTLInteraction]) -> None:
        mapping: dict[tuple[str, str], NRTLInteraction] = {}
        for interaction in interactions:
            key = (interaction.component_i, interaction.component_j)
            if key in mapping:
                raise ValueError(f"duplicate NRTL interaction {key!r}")
            mapping[key] = interaction
        self._interactions = mapping

    @property
    def interactions(self) -> Mapping[tuple[str, str], NRTLInteraction]:
        return self._interactions

    def missing_directed_pairs(self, species: Sequence[str]) -> tuple[tuple[str, str], ...]:
        keys = tuple(species)
        return tuple((i, j) for i in keys for j in keys
                     if i != j and (i, j) not in self._interactions)

    def activity_coefficients(self, *, temperature_k: float,
                              mole_fractions: Mapping[str, float]) -> Mapping[str, float]:
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        keys, x = _normalise(mole_fractions)
        n = len(keys)
        tau = np.zeros((n, n), dtype=float)
        alpha = np.zeros((n, n), dtype=float)
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                if i == j:
                    continue
                parameter = self._interactions.get((key_i, key_j))
                if parameter is not None:
                    tau[i, j] = parameter.delta_g_ij_j_mol / (R_GAS * temperature_k)
                    alpha[i, j] = parameter.alpha
        g = np.exp(np.clip(-alpha * tau, -700.0, 700.0))
        ln_gamma = np.zeros(n, dtype=float)
        for i in range(n):
            denominator_i = max(float(np.dot(x, g[:, i])), 1.0e-300)
            first = float(np.dot(x, tau[:, i] * g[:, i])) / denominator_i
            second = 0.0
            for j in range(n):
                denominator_j = max(float(np.dot(x, g[:, j])), 1.0e-300)
                weighted_tau = float(np.dot(x, tau[:, j] * g[:, j])) / denominator_j
                second += (x[j] * g[i, j] / denominator_j) * (tau[i, j] - weighted_tau)
            ln_gamma[i] = first + second
        gamma = np.exp(np.clip(ln_gamma, -700.0, 700.0))
        return {key: float(value) for key, value in zip(keys, gamma, strict=True)}

    def reduced_mixing_gibbs(self, *, temperature_k: float,
                             mole_fractions: Mapping[str, float]) -> float:
        keys, x = _normalise(mole_fractions)
        composition = {key: float(value) for key, value in zip(keys, x, strict=True)}
        gamma = self.activity_coefficients(
            temperature_k=temperature_k, mole_fractions=composition
        )
        return float(sum(
            x[index] * np.log(max(x[index] * float(gamma[key]), 1.0e-300))
            for index, key in enumerate(keys)
        ))


@dataclass(frozen=True, slots=True)
class LiquidPhaseSplitResult:
    phase_fractions_mol: tuple[float, ...]
    phase_compositions: tuple[Mapping[str, float], ...]
    single_phase_reduced_gibbs: float
    split_reduced_gibbs: float
    single_phase_stable: bool
    method: str


def select_activity_model(*, species: Sequence[str], database: ChemicalDatabase = BUILTIN_DATABASE,
                          mode: str = "auto") -> tuple[ActivityModel, tuple[str, ...]]:
    """Select an activity backend without disguising missing interaction data."""
    keys = tuple(dict.fromkeys(str(key) for key in species))
    if mode not in {"auto", "ideal", "nrtl"}:
        raise ValueError("activity model mode must be auto, ideal, or nrtl")
    if len(keys) < 2 or mode == "ideal":
        return IdealActivityModel(), ()
    nrtl = NRTLActivityModel(tuple(database.nrtl_interactions.values()))
    missing = nrtl.missing_directed_pairs(keys)
    if not missing:
        return nrtl, ()
    if mode == "nrtl":
        detail = ", ".join(f"{a}->{b}" for a, b in missing)
        return IdealActivityModel(), (
            f"NRTL requested but missing directed interaction parameters ({detail}); fell back to ideal",
        )
    return IdealActivityModel(), (
        "non-ideal binary interaction data incomplete for "
        + ", ".join(keys) + "; used ideal liquid activity model",
    )


def liquid_phase_stability(*, temperature_k: float, mole_fractions: Mapping[str, float],
                           activity_model: ActivityModel,
                           binary_grid_points: int = 241,
                           improvement_tolerance: float = 1.0e-8) -> LiquidPhaseSplitResult:
    """Test whether one liquid phase is stable and, when supported, split it.

    For binary mixtures a deterministic lower-envelope grid search is used. For
    larger mixtures a constrained two-phase Gibbs minimization is attempted. The
    standard-state contribution cancels under material balance, so the reduced
    mixing Gibbs energy is sufficient for comparing one and two liquid phases
    represented by the same activity model.
    """
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    keys, z_array = _normalise(mole_fractions)
    z = {key: float(value) for key, value in zip(keys, z_array, strict=True)}
    single = activity_model.reduced_mixing_gibbs(
        temperature_k=temperature_k, mole_fractions=z
    )
    if len(keys) < 2 or np.count_nonzero(z_array > _EPS) < 2:
        return LiquidPhaseSplitResult((1.0,), (z,), single, single, True, "single-component")

    if len(keys) == 2:
        points = max(int(binary_grid_points), 41)
        grid = np.linspace(1.0e-8, 1.0 - 1.0e-8, points)
        g_values = np.asarray([
            activity_model.reduced_mixing_gibbs(
                temperature_k=temperature_k,
                mole_fractions={keys[0]: float(value), keys[1]: float(1.0 - value)},
            )
            for value in grid
        ])
        overall = float(z_array[0])
        best_value = single
        best: tuple[float, float, float] | None = None
        for i, x_a in enumerate(grid[:-1]):
            if x_a > overall:
                break
            for j in range(i + 1, len(grid)):
                x_b = float(grid[j])
                if x_b < overall:
                    continue
                span = x_b - float(x_a)
                if span <= 0:
                    continue
                beta = (x_b - overall) / span
                if not 0 <= beta <= 1:
                    continue
                value = beta * float(g_values[i]) + (1.0 - beta) * float(g_values[j])
                if value < best_value:
                    best_value = value
                    best = (float(x_a), x_b, float(beta))
        if best is not None and best_value < single - improvement_tolerance:
            x_a, x_b, beta = best
            phase_a = {keys[0]: x_a, keys[1]: 1.0 - x_a}
            phase_b = {keys[0]: x_b, keys[1]: 1.0 - x_b}
            return LiquidPhaseSplitResult(
                (beta, 1.0 - beta), (phase_a, phase_b),
                single, best_value, False, "binary deterministic Gibbs lower envelope",
            )
        return LiquidPhaseSplitResult((1.0,), (z,), single, single, True,
                                      "binary deterministic Gibbs lower envelope")

    n = len(keys)

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        xa = np.maximum(vector[:n], 1.0e-12)
        xb = np.maximum(vector[n:2 * n], 1.0e-12)
        return xa, xb, float(vector[-1])

    def objective(vector: np.ndarray) -> float:
        xa, xb, beta = unpack(vector)
        ca = {key: float(value) for key, value in zip(keys, xa, strict=True)}
        cb = {key: float(value) for key, value in zip(keys, xb, strict=True)}
        return (
            beta * activity_model.reduced_mixing_gibbs(
                temperature_k=temperature_k, mole_fractions=ca
            )
            + (1.0 - beta) * activity_model.reduced_mixing_gibbs(
                temperature_k=temperature_k, mole_fractions=cb
            )
        )

    constraints = [
        {"type": "eq", "fun": lambda v: float(np.sum(v[:n]) - 1.0)},
        {"type": "eq", "fun": lambda v: float(np.sum(v[n:2 * n]) - 1.0)},
    ]
    for index in range(n):
        constraints.append({
            "type": "eq",
            "fun": lambda v, idx=index: float(
                v[-1] * v[idx] + (1.0 - v[-1]) * v[n + idx] - z_array[idx]
            ),
        })
    bounds = [(1.0e-12, 1.0)] * (2 * n) + [(1.0e-6, 1.0 - 1.0e-6)]
    starts: list[np.ndarray] = [
        np.concatenate((z_array, z_array, np.asarray([0.5]))),
    ]
    for index in range(min(n, 4)):
        xa = 0.15 * z_array
        xa[index] += 0.85
        xa /= xa.sum()
        xb = np.maximum(2.0 * z_array - xa, 1.0e-8)
        xb /= xb.sum()
        starts.append(np.concatenate((xa, xb, np.asarray([0.5]))))

    best_result = None
    for start in starts:
        result = minimize(
            objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1.0e-11, "maxiter": 1200},
        )
        if result.success and np.isfinite(result.fun):
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
    if best_result is not None and float(best_result.fun) < single - improvement_tolerance:
        xa, xb, beta = unpack(np.asarray(best_result.x, dtype=float))
        xa /= xa.sum()
        xb /= xb.sum()
        phase_a = {key: float(value) for key, value in zip(keys, xa, strict=True)}
        phase_b = {key: float(value) for key, value in zip(keys, xb, strict=True)}
        return LiquidPhaseSplitResult(
            (beta, 1.0 - beta), (phase_a, phase_b), single, float(best_result.fun),
            False, "constrained two-liquid Gibbs minimization",
        )
    return LiquidPhaseSplitResult((1.0,), (z,), single, single, True,
                                  "constrained two-liquid Gibbs minimization")


def liquid_mixture_density_kg_m3(*, species_mass_kg: Mapping[str, float],
                                 database: ChemicalDatabase = BUILTIN_DATABASE
                                 ) -> tuple[float | None, tuple[str, ...]]:
    """Ideal-volume mixture density used as an explicit fallback."""
    positive = {str(key): float(value) for key, value in species_mass_kg.items()
                if float(value) > 0}
    if not positive:
        return None, ()
    missing = tuple(sorted(
        key for key in positive
        if database.get(key).liquid_density_kg_m3 is None
        or database.get(key).liquid_density_kg_m3 <= 0
    ))
    if missing:
        return None, missing
    mass = sum(positive.values())
    volume = sum(
        value / float(database.get(key).liquid_density_kg_m3)
        for key, value in positive.items()
    )
    return (float(mass / volume) if volume > 0 else None), ()
