"""Feature-order-safe local model inference without a synthetic fallback."""
from __future__ import annotations

from datetime import datetime
from process_contract import MODES, normalize_params, with_constraints


def predict_local(condition, params, models):
    import numpy as np
    import pandas as pd

    p = normalize_params(condition, params)
    mode = MODES[condition]
    mods = models.get(mode, {})
    if any(mods.get(target) is None for target in ('cu', 'as', 'voltage')):
        raise RuntimeError('The three local surrogate models for this condition are unavailable. '
                           'Configure JCP_MODEL_DIR or launch the explicitly synthetic demo.')
    stamp = datetime.fromisoformat(str(params.get('timestamp', datetime.now().isoformat())))
    dates = {'year': stamp.year, 'month': stamp.month, 'day': stamp.day, 'hour': stamp.hour}
    features = {'电积前液Cu（g/L）': p['Cu_in'], **dates}
    prefixes = {'A': '三段', 'B': '四段'}
    units = ('A', 'B') if condition == 3 else (('A',) if condition == 1 else ('B',))
    vnames = ['电积前液Cu（g/L）']
    for u in units:
        prefix = prefixes[u]
        values = {prefix + '溶液温度℃': p[f'T_{u}'], prefix + '电流强度A': p[f'I_{u}'],
                  prefix + '流量m3/h': p[f'Q_{u}']}
        features.update(values)
        vnames.extend(values)
    vnames.extend(dates)

    def infer(target, order):
        model = mods[target]
        names = list(getattr(model, 'feature_names_in_', order))
        if set(names) != set(order) or len(names) != len(order):
            raise RuntimeError(f'The local {target} model feature schema does not match this condition')
        values = [[features[name] for name in names]]
        x = pd.DataFrame(values, columns=names) if hasattr(model, 'feature_names_in_') else np.array(values)
        value = float(model.predict(x)[0])
        if not np.isfinite(value):
            raise RuntimeError('The local model produced a non-finite prediction')
        return value

    voltage = infer('voltage', vnames)
    cnames = ['电积前液Cu（g/L）']
    for u in units:
        prefix = prefixes[u]
        features[prefix + '电压V'] = voltage
        cnames.extend([prefix + '溶液温度℃', prefix + '电流强度A',
                       prefix + '流量m3/h', prefix + '电压V'])
    cnames.extend(dates)
    power_name = 'total_power' if condition == 3 else ('three_stage_power' if condition == 1 else 'four_stage_power')
    features[power_name] = sum(p[f'I_{u}'] * voltage for u in units)
    cnames.append(power_name)
    cu, arsenic = infer('cu', cnames), infer('as', cnames)
    energy = features[power_name] * p['t'] / 1000
    flow = sum(p[f'Q_{u}'] for u in units) / len(units)
    copper = max(0.0, p['Cu_in'] - cu) * flow * p['t']
    profit = ((98.44 - 0.14 - 82.64) * copper - 2.5 * (11.64 + cu) * flow * p['t']
              - 0.6 * energy - 200 * p['t']) / 10000
    result = {'Cu_out': cu, 'As_out': arsenic, 'E': energy, 'profit': profit,
              'model_source': 'real', 'reference_time': stamp.isoformat(),
              'economic_basis': 'research accounting; series uses mean unit flow'}
    result.update({'V_A': voltage, 'V_B': voltage} if condition == 3 else {'V': voltage})
    return with_constraints(condition, p, result)
