"""Actual NSGA-II over the explicitly synthetic process evaluator."""
from __future__ import annotations

import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from process_contract import BOUNDS, demo_predict, normalize_params, result_row


def run_demo_nsga2(condition, params, *, population=64, generations=30, seed=42, crossover=0.9, eta=15, n_solutions=50):
    baseline = normalize_params(condition, params)
    # Incoming feed and the selected batch duration are not freely adjustable controls.
    keys = [key for key in BOUNDS[condition] if key not in ('Cu_in', 't')]
    bounds = [BOUNDS[condition][key] for key in keys]

    class DemoProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(keys), n_obj=4, n_ieq_constr=5,
                             xl=np.array([b[0] for b in bounds]), xu=np.array([b[1] for b in bounds]))

        def _evaluate(self, x, out, *args, **kwargs):
            p = {**baseline, **dict(zip(keys, map(float, x)))}
            r = demo_predict(condition, p)
            out['F'] = [r['Cu_out'], -r['As_out'], r['E'], -r['profit']]
            out['G'] = [((v['val'] - v['limit']) if v['operator'] == '<=' else
                         (v['limit'] - v['val'])) for v in r['constraints'].values()]

    start = time.perf_counter()
    result = minimize(DemoProblem(), NSGA2(pop_size=population,
                       crossover=SBX(prob=crossover, eta=eta),
                       mutation=PM(prob=1.0, prob_var=1 / len(keys), eta=20),
                       eliminate_duplicates=True), ('n_gen', generations), seed=seed, verbose=False)
    rows = []
    if result.X is not None:
        for x in np.atleast_2d(result.X):
            p = {**baseline, **dict(zip(keys, map(float, x)))}
            predicted = demo_predict(condition, p)
            if predicted['feasible']:
                rows.append(result_row(condition, p, predicted))
    if not rows:
        raise ValueError('No feasible candidate was found; change the input scenario or search budget')
    objectives = np.array([[r['Cu_out'], -r['As_out'], r['E'], -r['profit']] for r in rows])
    span = np.ptp(objectives, axis=0)
    scaled = (objectives - objectives.min(axis=0)) / np.where(span > 0, span, 1.0)
    indices = [int(np.argmin(objectives[:, 0])), int(np.argmin(objectives[:, 2])),
               int(np.argmin(objectives[:, 3])), int(np.argmin(scaled.mean(axis=1)))]
    scenes = ('Quality First', 'Energy First', 'Profit First', 'Balanced')
    selected = [{**rows[i], 'scene': scene} for scene, i in zip(scenes, indices)]
    retained = list(dict.fromkeys(indices))
    while len(retained) < min(n_solutions, len(rows)):
        distances = np.min(np.linalg.norm(scaled[:, None, :] - scaled[retained][None, :, :], axis=2), axis=1)
        distances[retained] = -1
        retained.append(int(np.argmax(distances)))
    return {'results': {f'cond{condition}': selected}, 'front': [rows[i] for i in retained],
            'baseline': demo_predict(condition, baseline), 'baseline_params': baseline,
            'full_front_size': len(rows),
            'runtime_seconds': time.perf_counter() - start,
            'evaluations': int(result.algorithm.evaluator.n_eval),
            'population': population, 'generations': generations, 'seed': seed,
            'model_source': 'synthetic_demo', 'algorithm': 'NSGA-II',
            'fixed_inputs': {'Cu_in': baseline['Cu_in'], 't': baseline['t']}}
