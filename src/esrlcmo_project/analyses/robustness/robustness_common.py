"""Shared, read-only access to the original ESRL-CMO and NSGA-II evaluators.

The robustness experiments use archived recommendations and call the original
surrogate, objective, economic, bound, and C1--C5 implementations.  No
optimizer or policy is retrained here.  Method-specific economic equations are
preserved exactly, including the archived Condition 3 flow convention; for
that reason Condition 3 absolute profit must not be used for a direct
cross-method economic ranking.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPT_ROOT = PROJECT_ROOT / "optimization"
RESULT_ROOT = OPT_ROOT / "result"

STATE_COLUMNS = {
    1: ["Cu_in", "TA", "IA", "QA", "t"],
    2: ["Cu_in", "TB", "IB", "QB", "t"],
    3: ["Cu_in", "TA", "IA", "QA", "TB", "IB", "QB", "t"],
}

ARCHIVE_PATHS = {
    "ESRL-CMO": {
        condition: RESULT_ROOT / f"outputs_condition{condition}"
        / f"pareto_ppo_condition{condition}.csv"
        for condition in (1, 2, 3)
    },
    "NSGA-II": {
        1: RESULT_ROOT / "newnsga2" / "pareto_nsga2_三段工况.csv",
        2: RESULT_ROOT / "newnsga2" / "pareto_nsga2_四段工况.csv",
        3: RESULT_ROOT / "newnsga2" / "pareto_nsga2_串联工况.csv",
    },
}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_archive(method: str, condition: int) -> pd.DataFrame:
    path = ARCHIVE_PATHS[method][condition]
    frame = pd.read_csv(path)
    required = set(STATE_COLUMNS[condition]) | {"Cu_out", "As_out", "E_total", "Net_profit"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).copy()
    frame["source_file"] = str(path)
    return frame


def select_representatives(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministic maximin coverage of the physical archived outcome space.

    Cu_out, -As_out and log1p(E_total) are used.  Profit is intentionally not
    used because archived NSGA-II and ESRL-CMO profit units/conventions are not
    identical in every condition.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    frame = frame.reset_index(drop=True)
    if len(frame) <= n:
        return frame.copy()
    raw = np.column_stack([
        frame["Cu_out"].to_numpy(float),
        -frame["As_out"].to_numpy(float),
        np.log1p(np.maximum(frame["E_total"].to_numpy(float), 0.0)),
    ])
    lo, hi = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    scaled = (raw - lo) / np.where(hi > lo, hi - lo, 1.0)
    selected = [int(np.argmin(np.linalg.norm(scaled, axis=1)))]
    min_distance = np.linalg.norm(scaled - scaled[selected[0]], axis=1)
    min_distance[selected[0]] = -np.inf
    while len(selected) < n:
        idx = int(np.argmax(min_distance))
        selected.append(idx)
        distance = np.linalg.norm(scaled - scaled[idx], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    result = frame.iloc[selected].copy().reset_index(drop=True)
    result.insert(0, "representative_id", np.arange(1, len(result) + 1))
    return result


class OriginalEvaluators:
    """Evaluate states with each method's untouched original implementation."""

    def __init__(self, runtime: dict[str, int]):
        self.runtime = dict(runtime)
        self.nsga = _load_module(
            "esrl_cmo_original_nsga", OPT_ROOT / "nsga2_optimization.py"
        )
        self.nsga.surrogate = self.nsga.SurrogateModel()
        for target_models in self.nsga.surrogate._models.values():
            for model in target_models.values():
                if hasattr(model, "n_jobs"):
                    model.n_jobs = 1
        self.ppo = _load_module(
            "esrl_cmo_original_ppo", OPT_ROOT / "ppo_lagrangian.py"
        )
        self._nsga_problems: dict[int, object] = {}
        self._ppo_models: dict[int, object] = {}

    def bounds(self, method: str, condition: int) -> tuple[np.ndarray, np.ndarray]:
        if method == "NSGA-II":
            problem = self._nsga_problem(condition)
            return np.asarray(problem.xl, float), np.asarray(problem.xu, float)
        self.ppo.make_cfg(condition, fast=False)
        return np.asarray(self.ppo._S_LO, float), np.asarray(self.ppo._S_HI, float)

    def _nsga_problem(self, condition: int):
        if condition not in self._nsga_problems:
            name = {
                1: "ThreeStageOptProblem",
                2: "FourStageOptProblem",
                3: "SerialOptProblem",
            }[condition]
            self._nsga_problems[condition] = getattr(self.nsga, name)(self.runtime)
        return self._nsga_problems[condition]

    def _ppo_model(self, condition: int):
        if condition not in self._ppo_models:
            self.ppo.make_cfg(condition, fast=False)
            self.ppo.CFG["device"] = "cpu"
            self._ppo_models[condition] = self.ppo.load_models(
                str(OPT_ROOT / "joblib"), n_jobs=1
            )
        return self._ppo_models[condition]

    def evaluate(self, method: str, condition: int, states: np.ndarray) -> dict[str, np.ndarray]:
        states = np.asarray(states, dtype=float)
        low, high = self.bounds(method, condition)
        states = np.clip(states, low, high)
        if method == "NSGA-II":
            out: dict[str, np.ndarray] = {}
            self._nsga_problem(condition)._evaluate(states, out)
            f = np.asarray(out["F"], float)
            g = np.asarray(out["G"], float)
            return {
                "Cu_out": f[:, 0],
                "As_out": -f[:, 1],
                "E_total": f[:, 2],
                "Net_profit_10k_CNY": -f[:, 3] / 1.0e4,
                "constraint_violation": np.maximum(g, 0.0),
                "constraint_violated": g > 1.0e-8,
                "feasible": ~(g > 1.0e-8).any(axis=1),
                "feasibility_tolerance": np.full(len(g), 1.0e-8),
            }

        self.ppo.make_cfg(condition, fast=False)
        models = self._ppo_model(condition)
        rt = self.runtime
        if condition in (1, 2):
            cu, arsenic, voltage = self.ppo.predict_all_batch(
                models, *(states[:, i] for i in range(5)), **rt
            )
            energy, profit = self.ppo.compute_objectives_batch(
                tuple(states[:, i] for i in range(5)), cu, arsenic, voltage
            )
            costs = self.ppo.compute_costs_batch(cu, arsenic, states[:, 2], voltage)
        else:
            cu, arsenic, _vmean, va, vb = self.ppo.predict_all_batch(
                models, *(states[:, i] for i in range(8)), **rt
            )
            energy, profit = self.ppo.compute_objectives_batch(
                tuple(states[:, i] for i in range(8)), cu, arsenic, (_vmean, va, vb)
            )
            costs = self.ppo.compute_costs_batch(
                cu, arsenic, (states[:, 2], states[:, 5]), (va, vb)
            )
        costs = np.asarray(costs, float)
        return {
            "Cu_out": np.asarray(cu, float),
            "As_out": np.asarray(arsenic, float),
            "E_total": np.asarray(energy, float),
            "Net_profit_10k_CNY": np.asarray(profit, float),
            "constraint_violation": costs,
            "constraint_violated": costs >= 1.0e-3,
            "feasible": ~(costs >= 1.0e-3).any(axis=1),
            "feasibility_tolerance": np.full(len(costs), 1.0e-3),
        }


def state_matrix(frame: pd.DataFrame, condition: int) -> np.ndarray:
    return frame[STATE_COLUMNS[condition]].to_numpy(dtype=float)


def scenario_statistics(evaluated: dict[str, np.ndarray],
                        baseline: dict[str, np.ndarray]) -> dict[str, float]:
    row: dict[str, float] = {
        "feasibility_rate": float(np.mean(evaluated["feasible"])),
    }
    violations = evaluated["constraint_violated"]
    for i in range(5):
        row[f"C{i + 1}_violation_rate"] = float(np.mean(violations[:, i]))
    for name in ("Cu_out", "As_out", "E_total", "Net_profit_10k_CNY"):
        delta = np.asarray(evaluated[name]) - np.asarray(baseline[name])
        row[f"{name}_mean"] = float(np.mean(evaluated[name]))
        row[f"{name}_mean_shift"] = float(np.mean(delta))
        row[f"{name}_mean_absolute_shift"] = float(np.mean(np.abs(delta)))
        row[f"{name}_p95_absolute_shift"] = float(np.percentile(np.abs(delta), 95))
    return row
