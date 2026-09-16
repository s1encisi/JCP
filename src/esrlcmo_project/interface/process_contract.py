"""Shared process input, objective, constraint and policy-action contracts.

The synthetic evaluator is an explicitly labelled software demonstration. It
is not a fitted surrogate, a mechanistic plant model, or experimental evidence.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

MODES = {1: 'three_stage', 2: 'four_stage', 3: 'serial'}
CATHODE_AREA = 35 * 1.100 * 1.029 * 2
LIMITS = {'Cu_out': 8.0, 'As_out': 4.0, 'J': 340.0, 'V_cell': 2.5, 'Cu_As': 0.8}
BOUNDS = {
    1: {'Cu_in': (29, 55), 'T_A': (48, 65), 'I_A': (8000, 27000),
        'Q_A': (111, 123), 't': (2, 8)},
    2: {'Cu_in': (29, 55), 'T_B': (55, 65), 'I_B': (8000, 27000),
        'Q_B': (113, 123), 't': (2, 8)},
    3: {'Cu_in': (29, 55), 'T_A': (40, 65), 'I_A': (8000, 27000),
        'Q_A': (111, 123), 'T_B': (40, 65), 'I_B': (8000, 27000),
        'Q_B': (113, 123), 't': (2, 8)},
}
DEFAULTS = {
    1: {'Cu_in': 38.0, 'T_A': 58.0, 'I_A': 18000.0, 'Q_A': 117.0, 't': 4.0},
    2: {'Cu_in': 38.0, 'T_B': 60.0, 'I_B': 18000.0, 'Q_B': 118.0, 't': 4.0},
    3: {'Cu_in': 38.0, 'T_A': 56.0, 'I_A': 17000.0, 'Q_A': 117.0,
        'T_B': 59.0, 'I_B': 16000.0, 'Q_B': 118.0, 't': 6.0},
}
STATE_KEYS = {condition: tuple(bounds) for condition, bounds in BOUNDS.items()}


def condition_number(value) -> int:
    if isinstance(value, bool) or str(value) not in ('1', '2', '3'):
        raise ValueError('condition must be 1, 2, or 3')
    return int(value)


def finite_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{field} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be a finite number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{field} must be a finite number')
    return number


def normalize_params(condition, values: Mapping, *, allow_defaults=False) -> dict:
    condition = condition_number(condition)
    if not isinstance(values, Mapping):
        raise ValueError('params must be an object')
    supplied = dict(values)
    aliases = {'Cu_in': ('cu_in',), 't': ('duration',)}
    for unit in ('A', 'B'):
        for prefix, long_name in [('T', 'temperature'), ('I', 'current'), ('Q', 'flow')]:
            key = f'{prefix}_{unit}'
            aliases[key] = (f'{prefix}{unit}',)
            if condition != 3:
                aliases[key] += (prefix, long_name)
    normalized = {}
    for key, (low, high) in BOUNDS[condition].items():
        value = supplied.get(key)
        if value is None:
            value = next((supplied[a] for a in aliases.get(key, ()) if a in supplied), None)
        if value is None and allow_defaults:
            value = DEFAULTS[condition][key]
        if value is None:
            raise ValueError(f'Missing process parameter: {key}')
        value = finite_number(value, key)
        if not low <= value <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
        normalized[key] = value
    return normalized


def check_constraints(condition, params: Mapping, prediction: Mapping) -> dict:
    condition = condition_number(condition)
    cu = finite_number(prediction['Cu_out'], 'Cu_out')
    arsenic = finite_number(prediction['As_out'], 'As_out')
    units = ('A', 'B') if condition == 3 else (('A',) if condition == 1 else ('B',))
    currents = [finite_number(params[f'I_{u}'], f'I_{u}') for u in units]
    if condition == 3:
        voltages = [finite_number(prediction[f'V_{u}'], f'V_{u}') for u in units]
    else:
        voltages = [finite_number(prediction['V'], 'V')]
    density, cell_voltage = max(currents) / CATHODE_AREA, max(voltages) / 16
    ratio = cu / arsenic if arsenic > 0 else None
    values = [('C1_Cu', cu, '<=', 8.0), ('C2_As', arsenic, '>=', 4.0),
              ('C3_J', density, '<=', 340.0), ('C4_V', cell_voltage, '<=', 2.5),
              ('C5_CuAs', ratio, '<=', 0.8)]
    return {key: {'val': value, 'operator': operator, 'limit': limit,
                  'ok': value is not None and (value <= limit if operator == '<=' else value >= limit)}
            for key, value, operator, limit in values}


def with_constraints(condition, params: Mapping, prediction: Mapping) -> dict:
    result = dict(prediction)
    for key in ('Cu_out', 'As_out', 'E', 'profit'):
        finite_number(result[key], key)
    if any(result[key] < 0 for key in ('Cu_out', 'As_out', 'E')):
        raise RuntimeError('Model concentrations and energy must be non-negative')
    result['condition'] = condition_number(condition)
    result['constraints'] = check_constraints(condition, params, result)
    result['feasible'] = all(item['ok'] for item in result['constraints'].values())
    return result


def demo_predict(condition, params: Mapping) -> dict:
    """Deterministic synthetic formula; the coefficients are demo assumptions."""
    condition = condition_number(condition)
    p = normalize_params(condition, params)
    units = ('A', 'B') if condition == 3 else (('A',) if condition == 1 else ('B',))
    currents = [p[f'I_{u}'] for u in units]
    temperature = sum(p[f'T_{u}'] for u in units) / len(units)
    flow = sum(p[f'Q_{u}'] for u in units) / len(units)
    current_effect = sum(i - 8000 for i in currents) * (0.00013 if condition == 3 else 0.00020)
    cu = max(1.0, p['Cu_in'] * 0.18 - current_effect - 0.20 * (p['t'] - 2)
             - 0.018 * (temperature - 45) + 0.02 * (flow - 115))
    arsenic = 6.3 + 0.045 * (flow - 115) - 0.000035 * (sum(currents)/len(units) - 15000)
    voltages = {u: 16 * (1.65 + 0.000025 * (p[f'I_{u}'] - 8000)
                        - 0.006 * (p[f'T_{u}'] - 45)) for u in units}
    energy = sum(voltages[u] * p[f'I_{u}'] for u in units) * p['t'] / 1000
    recovered = (p['Cu_in'] - cu) * flow * p['t']
    profit = (8.0 * recovered - 0.65 * energy - 150 * p['t']) / 10000
    result = {'Cu_out': cu, 'As_out': arsenic, 'E': energy, 'profit': profit,
              'model_source': 'synthetic_demo', 'economic_basis': 'synthetic demonstration coefficients'}
    if condition == 3:
        result.update(V_A=voltages['A'], V_B=voltages['B'])
    else:
        result['V'] = next(iter(voltages.values()))
    return with_constraints(condition, p, result)


def decode_policy_action(condition, params: Mapping, action: Sequence[float], *, fixed_feed=True) -> dict:
    """Decode trained [dCu, dT, dI, dQ, ...] increments, never absolute setpoints."""
    condition = condition_number(condition)
    p = normalize_params(condition, params)
    keys = STATE_KEYS[condition][:-1]
    scales = (2.0, 2.0, 1000.0, 1.0) if condition != 3 else (2.0, 2.0, 1000.0, 1.0, 2.0, 1000.0, 1.0)
    if len(action) != len(keys):
        raise ValueError('Policy action dimensions do not match the operating condition')
    result = dict(p)
    for key, scale, raw in zip(keys, scales, action):
        a = finite_number(raw, 'policy action')
        if not -1.00001 <= a <= 1.00001:
            raise ValueError('Policy action is outside its trained normalized range')
        if key == 'Cu_in' and fixed_feed:
            continue
        low, high = BOUNDS[condition][key]
        result[key] = min(high, max(low, p[key] + max(-1, min(1, a)) * scale))
    return result


def result_row(condition, params: Mapping, prediction: Mapping, scene='Candidate') -> dict:
    result = {**params, **prediction, 'scene': scene}
    for u in ('A', 'B'):
        for key in ('T', 'I', 'Q'):
            if f'{key}_{u}' in params:
                result[f'{key}{u}'] = params[f'{key}_{u}']
    if condition in (1, 2):
        u = 'A' if condition == 1 else 'B'
        for key in ('T', 'I', 'Q'):
            result[key] = params[f'{key}_{u}']
    return result
