"""Preliminary multi-objective algorithm screening for the ESRL-CMO problem.

This script does not modify the original ESRL-CMO source tree.  It reuses the
copied surrogate-assisted optimization problem in ``nsga2_optimization.py``
and screens ten methods in total: a matched NSGA-II reference and nine
non-NSGA-III candidates.

The default configuration is deliberately a screening budget (100 population
members x 100 generations, three seeds and all three operating conditions).
It is intended to shortlist algorithms before any publication-budget rerun.
No optimization is started unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

import constrained_moea_extensions as base


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "moea_algorithm_screening"
REFERENCE_NSGA2_ROOT = ROOT / "constrained_moea_benchmark" / "nsga2"

ALGORITHMS = (
    "nsga2",
    "spea2",
    "ctaea",
    "agemoea",
    "agemoea2",
    "rvea",
    "smsemoa",
    "rnsga2",
    "moead_tch_batch",
    "moead_pbi_batch",
)

LABELS = {
    "nsga2": "NSGA-II (screening reference)",
    "spea2": "SPEA2",
    "ctaea": "C-TAEA",
    "agemoea": "AGE-MOEA",
    "agemoea2": "AGE-MOEA2",
    "rvea": "RVEA",
    "smsemoa": "SMS-EMOA",
    "rnsga2": "R-NSGA-II",
    "moead_tch_batch": "MOEA/D-TCH (batch)",
    "moead_pbi_batch": "MOEA/D-PBI (batch)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=[*ALGORITHMS, "all"], default="all")
    parser.add_argument("--condition", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 89])
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--day", type=int, default=15)
    parser.add_argument("--hour", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["Cu_out"].to_numpy(float),
            -frame["As_out"].to_numpy(float),
            frame["E_total"].to_numpy(float),
            -frame["Net_profit"].to_numpy(float),
        ]
    )


def maximin_indices(values: np.ndarray, n_select: int) -> np.ndarray:
    """Select deterministic, well-spread representatives in normalized space."""
    n_select = min(int(n_select), len(values))
    if n_select <= 0:
        return np.empty(0, dtype=int)
    ideal = values.min(axis=0)
    span = np.where(values.max(axis=0) > ideal, values.max(axis=0) - ideal, 1.0)
    normalized = (values - ideal) / span
    selected: list[int] = []
    for j in range(normalized.shape[1]):
        idx = int(np.argmin(normalized[:, j]))
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= n_select:
            return np.asarray(selected[:n_select], dtype=int)
    while len(selected) < n_select:
        chosen = normalized[np.asarray(selected)]
        distances = np.linalg.norm(normalized[:, None, :] - chosen[None, :, :], axis=2)
        nearest = distances.min(axis=1)
        nearest[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return np.asarray(selected, dtype=int)


def rnsga2_reference_points(condition: int, n_points: int = 10) -> np.ndarray:
    files = sorted(
        REFERENCE_NSGA2_ROOT.glob(f"seed_*/pareto_nsga2_condition{condition}.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"Matched NSGA-II reference fronts are required for R-NSGA-II: {REFERENCE_NSGA2_ROOT}"
        )
    pooled = np.vstack([objective_matrix(pd.read_csv(path)) for path in files])
    finite = np.isfinite(pooled).all(axis=1)
    pooled = pooled[finite]
    nd = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
    front = pooled[nd]
    return front[maximin_indices(front, n_points)]


def common_operators(problem) -> dict:
    return {
        "sampling": FloatRandomSampling(),
        "crossover": SBX(prob=0.9, eta=15, n_offsprings=2),
        "mutation": PM(prob=1.0, prob_var=1.0 / problem.n_var, eta=20),
    }


def bounded_sbx_and_pm(
    parents_a: np.ndarray,
    parents_b: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
    crossover_probability: float = 0.9,
    crossover_eta: float = 15.0,
    mutation_eta: float = 20.0,
) -> np.ndarray:
    """Generate one bounded SBX child per parent pair, then apply polynomial mutation."""
    children = parents_a.copy()
    n_children, n_var = children.shape
    do_crossover = rng.random(n_children) < crossover_probability
    for i in np.flatnonzero(do_crossover):
        for j in range(n_var):
            if rng.random() > 0.5:
                continue
            p1, p2 = float(parents_a[i, j]), float(parents_b[i, j])
            if abs(p1 - p2) <= 1.0e-14:
                continue
            y1, y2 = min(p1, p2), max(p1, p2)
            rand = rng.random()
            beta = 1.0 + 2.0 * (y1 - xl[j]) / (y2 - y1)
            alpha = 2.0 - beta ** (-(crossover_eta + 1.0))
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (crossover_eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (
                    1.0 / (crossover_eta + 1.0)
                )
            child_low = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
            beta = 1.0 + 2.0 * (xu[j] - y2) / (y2 - y1)
            alpha = 2.0 - beta ** (-(crossover_eta + 1.0))
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (crossover_eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (
                    1.0 / (crossover_eta + 1.0)
                )
            child_high = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
            children[i, j] = child_low if rng.random() < 0.5 else child_high

    mutation_probability = 1.0 / n_var
    for i in range(n_children):
        for j in range(n_var):
            if rng.random() >= mutation_probability:
                continue
            y = float(np.clip(children[i, j], xl[j], xu[j]))
            delta1 = (y - xl[j]) / (xu[j] - xl[j])
            delta2 = (xu[j] - y) / (xu[j] - xl[j])
            rand = rng.random()
            mut_pow = 1.0 / (mutation_eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (mutation_eta + 1.0)
                delta_q = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = (
                    2.0 * (1.0 - rand)
                    + 2.0 * (rand - 0.5) * xy ** (mutation_eta + 1.0)
                )
                delta_q = 1.0 - val**mut_pow
            children[i, j] = y + delta_q * (xu[j] - xl[j])
    return np.clip(children, xl, xu)


def batch_moead(
    problem,
    pop_size: int,
    generations: int,
    seed: int,
    decomposition: str,
) -> tuple[np.ndarray, int, dict]:
    """A generational, batch-evaluated MOEA/D for expensive vectorized surrogates."""
    rng = np.random.default_rng(seed)
    ref_dirs = get_reference_directions("energy", problem.n_obj, pop_size, seed=seed)
    if len(ref_dirs) != pop_size:
        raise RuntimeError(f"Expected {pop_size} reference directions, received {len(ref_dirs)}")
    distances = np.linalg.norm(ref_dirs[:, None, :] - ref_dirs[None, :, :], axis=2)
    n_neighbors = min(20, pop_size)
    neighbors = np.argsort(distances, axis=1)[:, :n_neighbors]
    xl, xu = np.asarray(problem.xl, float), np.asarray(problem.xu, float)
    x = rng.uniform(xl, xu, size=(pop_size, problem.n_var))
    f, g = problem.evaluate(x, return_values_of=["F", "G"])
    f, g = np.asarray(f, float), np.asarray(g, float)
    n_eval = pop_size

    for _ in range(1, generations):
        parent_a = np.empty(pop_size, dtype=int)
        parent_b = np.empty(pop_size, dtype=int)
        all_indices = np.arange(pop_size)
        for k in range(pop_size):
            pool = neighbors[k] if rng.random() < 0.9 else all_indices
            chosen = rng.choice(pool, size=2, replace=len(pool) < 2)
            parent_a[k], parent_b[k] = int(chosen[0]), int(chosen[1])
        offspring_x = bounded_sbx_and_pm(x[parent_a], x[parent_b], xl, xu, rng)
        offspring_f, offspring_g = problem.evaluate(
            offspring_x, return_values_of=["F", "G"]
        )
        offspring_f = np.asarray(offspring_f, float)
        offspring_g = np.asarray(offspring_g, float)
        n_eval += pop_size

        feasible_pool = np.vstack([f, offspring_f])[
            np.vstack([g, offspring_g]).max(axis=1) <= 1.0e-8
        ]
        scale_pool = feasible_pool if len(feasible_pool) else np.vstack([f, offspring_f])
        ideal = scale_pool.min(axis=0)
        span = np.where(scale_pool.max(axis=0) > ideal, scale_pool.max(axis=0) - ideal, 1.0)

        for k in rng.permutation(pop_size):
            candidate_f = offspring_f[k]
            candidate_g = offspring_g[k]
            candidate_cv = float(np.maximum(candidate_g, 0.0).sum())
            indices = neighbors[k]
            current_cv = np.maximum(g[indices], 0.0).sum(axis=1)
            candidate_feasible = candidate_cv <= 1.0e-8
            current_feasible = current_cv <= 1.0e-8
            replace = candidate_feasible & np.logical_not(current_feasible)
            replace |= (not candidate_feasible) & np.logical_not(current_feasible) & (
                candidate_cv < current_cv
            )
            both_feasible = candidate_feasible & current_feasible
            if np.any(both_feasible):
                weights = np.maximum(ref_dirs[indices], 1.0e-8)
                current_norm = (f[indices] - ideal) / span
                candidate_norm = (candidate_f - ideal) / span
                if decomposition == "tchebycheff":
                    current_value = np.max(weights * np.abs(current_norm), axis=1)
                    candidate_value = np.max(weights * np.abs(candidate_norm), axis=1)
                elif decomposition == "pbi":
                    unit = weights / np.linalg.norm(weights, axis=1, keepdims=True)
                    current_d1 = np.sum(current_norm * unit, axis=1)
                    candidate_d1 = np.sum(candidate_norm * unit, axis=1)
                    current_d2 = np.linalg.norm(current_norm - current_d1[:, None] * unit, axis=1)
                    candidate_d2 = np.linalg.norm(candidate_norm - candidate_d1[:, None] * unit, axis=1)
                    current_value = current_d1 + 5.0 * current_d2
                    candidate_value = candidate_d1 + 5.0 * candidate_d2
                else:
                    raise KeyError(decomposition)
                replace |= both_feasible & (candidate_value < current_value)
            replace_indices = indices[np.flatnonzero(replace)]
            if len(replace_indices) > 2:
                replace_indices = rng.choice(replace_indices, size=2, replace=False)
            if len(replace_indices):
                x[replace_indices] = offspring_x[k]
                f[replace_indices] = candidate_f
                g[replace_indices] = candidate_g

    metadata = {
        "constraint_handling": (
            "feasibility-first neighborhood replacement using the five original constraints"
        ),
        "optimizer_objectives": problem.n_obj,
        "reference_structure": (
            f"{len(ref_dirs)} energy reference directions; 20-neighbor batch MOEA/D; "
            f"{decomposition} decomposition"
        ),
    }
    return x, n_eval, metadata


def build_algorithm(name: str, original_problem, condition: int, pop_size: int, seed: int):
    """Return algorithm, optimizer-facing problem, and method metadata."""
    common = common_operators(original_problem)
    constrained_problem = original_problem
    metadata = {
        "constraint_handling": "native feasibility-first",
        "optimizer_objectives": original_problem.n_obj,
        "reference_structure": "not applicable",
    }

    if name == "nsga2":
        algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True, **common)
    elif name == "spea2":
        algorithm = SPEA2(pop_size=pop_size, eliminate_duplicates=True, **common)
    elif name == "ctaea":
        ref_dirs = get_reference_directions("energy", original_problem.n_obj, pop_size, seed=seed)
        algorithm = CTAEA(ref_dirs=ref_dirs, eliminate_duplicates=True, **common)
        metadata["reference_structure"] = f"{len(ref_dirs)} energy reference directions"
    elif name == "agemoea":
        algorithm = AGEMOEA(pop_size=pop_size, eliminate_duplicates=True, **common)
    elif name == "agemoea2":
        algorithm = AGEMOEA2(pop_size=pop_size, eliminate_duplicates=True, **common)
    elif name == "rvea":
        ref_dirs = get_reference_directions("energy", original_problem.n_obj, pop_size, seed=seed)
        algorithm = RVEA(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            eliminate_duplicates=True,
            **common,
        )
        metadata["reference_structure"] = f"{len(ref_dirs)} energy reference vectors"
    elif name == "smsemoa":
        algorithm = SMSEMOA(
            pop_size=pop_size,
            normalize=True,
            eliminate_duplicates=True,
            **common,
        )
    elif name == "rnsga2":
        ref_points = rnsga2_reference_points(condition, n_points=min(10, pop_size))
        algorithm = RNSGA2(
            ref_points=ref_points,
            epsilon=0.01,
            normalization="front",
            pop_size=pop_size,
            eliminate_duplicates=True,
            **common,
        )
        metadata["reference_structure"] = (
            f"{len(ref_points)} deterministic points from matched NSGA-II fronts"
        )
    else:
        raise KeyError(name)

    return algorithm, constrained_problem, metadata


def evaluate_x_population(original_problem, x_all: np.ndarray):
    if x_all is None or len(x_all) == 0:
        return (
            np.empty((0, original_problem.n_obj)),
            np.empty((0, original_problem.n_var)),
            np.empty((0, original_problem.n_ieq_constr)),
            0.0,
        )
    x_all = np.asarray(x_all, dtype=float)
    f_all, g_all = original_problem.evaluate(x_all, return_values_of=["F", "G"])
    f_all = np.asarray(f_all, dtype=float)
    g_all = np.asarray(g_all, dtype=float)
    finite = np.isfinite(x_all).all(axis=1) & np.isfinite(f_all).all(axis=1)
    feasible = finite & (g_all <= 1.0e-8).all(axis=1)
    feasible_rate = float(feasible.mean()) if len(feasible) else 0.0
    f, x, g = f_all[feasible], x_all[feasible], g_all[feasible]
    if len(f):
        nd = NonDominatedSorting().do(f, only_non_dominated_front=True)
        f, x, g = f[nd], x[nd], g[nd]
    return f, x, g, feasible_rate


def evaluate_final_population(original_problem, result):
    if result.pop is None:
        return evaluate_x_population(original_problem, np.empty((0, original_problem.n_var)))
    return evaluate_x_population(original_problem, result.pop.get("X"))


def write_metadata(path: Path, row: dict) -> None:
    current = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path)
        current = pd.concat([previous, current], ignore_index=True)
        current = current.drop_duplicates(
            subset=["algorithm", "condition", "seed"], keep="last"
        )
    current.to_csv(path, index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    if args.population_size < 10 or args.generations < 1:
        raise ValueError("population-size must be >= 10 and generations must be >= 1")
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required")
    algorithms = list(ALGORITHMS if args.algorithm == "all" else [args.algorithm])
    conditions = [1, 2, 3] if args.condition == "all" else [int(args.condition)]
    runtime = {
        "year": int(args.year),
        "month": int(args.month),
        "day": int(args.day),
        "hour": int(args.hour),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "run_metadata.csv"
    config = {
        "purpose": "preliminary algorithm screening; not publication-budget final runs",
        "algorithms": algorithms,
        "labels": {name: LABELS[name] for name in algorithms},
        "conditions": conditions,
        "seeds": seeds,
        "population_size": args.population_size,
        "generations": args.generations,
        "nominal_evaluations_per_run": args.population_size * args.generations,
        "runtime_features": runtime,
        "bounds_domain": "archived ESRL-CMO domains",
        "source_problem": str(base.SOURCE),
        "objective_order": ["Cu_out", "-As_out", "E_total", "-Net_profit"],
        "constraints": "five original G <= 0 constraints",
        "common_operators": {
            "sampling": "FloatRandomSampling",
            "crossover": "SBX(prob=0.9, eta=15, n_offsprings=2)",
            "mutation": "PM(prob=1.0, prob_var=1/n_var, eta=20)",
        },
    }
    (args.output_dir / "screening_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    module = base.load_source_module()
    module.surrogate = module.SurrogateModel()
    for target_models in module.surrogate._models.values():
        for model in target_models.values():
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1

    total = len(algorithms) * len(conditions) * len(seeds)
    run_index = 0
    for condition in conditions:
        class_name, condition_label, variable_names = base.CONDITION_DEFS[condition]
        problem = getattr(module, class_name)(runtime)
        xl, xu = base.ESRLCMO_BOUNDS[condition]
        problem.xl = np.asarray(xl, dtype=float)
        problem.xu = np.asarray(xu, dtype=float)
        for algorithm_name in algorithms:
            for seed in seeds:
                run_index += 1
                run_dir = args.output_dir / algorithm_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                front_path = run_dir / f"pareto_{algorithm_name}_condition{condition}.csv"
                if front_path.exists() and not args.overwrite:
                    print(
                        f"[{run_index}/{total}] skip existing {algorithm_name}, "
                        f"Condition {condition}, seed {seed}",
                        flush=True,
                    )
                    continue

                print(
                    f"[{run_index}/{total}] run {algorithm_name}, "
                    f"Condition {condition}, seed {seed}",
                    flush=True,
                )
                started = time.perf_counter()
                try:
                    if algorithm_name in {"moead_tch_batch", "moead_pbi_batch"}:
                        decomposition = (
                            "tchebycheff"
                            if algorithm_name == "moead_tch_batch"
                            else "pbi"
                        )
                        final_x, n_evaluations, method_metadata = batch_moead(
                            problem,
                            args.population_size,
                            args.generations,
                            seed,
                            decomposition,
                        )
                        f, x, g, feasible_rate = evaluate_x_population(problem, final_x)
                    else:
                        algorithm, run_problem, method_metadata = build_algorithm(
                            algorithm_name, problem, condition, args.population_size, seed
                        )
                        result = minimize(
                            run_problem,
                            algorithm,
                            ("n_gen", args.generations),
                            seed=seed,
                            verbose=False,
                            save_history=False,
                        )
                        n_evaluations = int(result.algorithm.evaluator.n_eval)
                        f, x, g, feasible_rate = evaluate_final_population(problem, result)
                    elapsed = time.perf_counter() - started
                    table = base.result_table(
                        f,
                        x,
                        g,
                        LABELS[algorithm_name],
                        condition_label,
                        variable_names,
                    )
                    table.to_csv(front_path, index=False, encoding="utf-8-sig")
                    row = {
                        "status": "completed",
                        "algorithm": algorithm_name,
                        "algorithm_label": LABELS[algorithm_name],
                        "condition": condition_label,
                        "seed": seed,
                        "population_size": args.population_size,
                        "generations": args.generations,
                        "n_evaluations": n_evaluations,
                        "runtime_seconds": elapsed,
                        "final_population_feasibility_rate": feasible_rate,
                        "n_final_feasible_nondominated": len(table),
                        "bounds_domain": "esrlcmo",
                        "runtime_features": json.dumps(runtime),
                        "error_type": "",
                        "error_message": "",
                        **method_metadata,
                    }
                    print(
                        f"    completed in {elapsed:.2f}s; feasible={feasible_rate:.3f}; "
                        f"ND={len(table)}",
                        flush=True,
                    )
                except Exception as exc:  # keep the screening auditable
                    elapsed = time.perf_counter() - started
                    row = {
                        "status": "failed",
                        "algorithm": algorithm_name,
                        "algorithm_label": LABELS[algorithm_name],
                        "condition": condition_label,
                        "seed": seed,
                        "population_size": args.population_size,
                        "generations": args.generations,
                        "n_evaluations": 0,
                        "runtime_seconds": elapsed,
                        "final_population_feasibility_rate": np.nan,
                        "n_final_feasible_nondominated": 0,
                        "bounds_domain": "esrlcmo",
                        "runtime_features": json.dumps(runtime),
                        "constraint_handling": "not completed",
                        "optimizer_objectives": np.nan,
                        "reference_structure": "not completed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    print(
                        f"    FAILED after {elapsed:.2f}s: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                write_metadata(metadata_path, row)


def print_plan(args: argparse.Namespace) -> None:
    algorithms = list(ALGORITHMS if args.algorithm == "all" else [args.algorithm])
    conditions = [1, 2, 3] if args.condition == "all" else [int(args.condition)]
    print(
        json.dumps(
            {
                "status": "NOT RUN - supply --execute to start",
                "algorithms": algorithms,
                "NSGA_III_included": False,
                "conditions": conditions,
                "seeds": args.seeds,
                "population_size": args.population_size,
                "generations": args.generations,
                "nominal_evaluations_per_run": args.population_size * args.generations,
                "planned_runs": len(algorithms) * len(conditions) * len(set(args.seeds)),
                "output": str(args.output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    if not args.execute:
        print_plan(args)
        return
    run(args)


if __name__ == "__main__":
    main()
