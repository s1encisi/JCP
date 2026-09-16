"""
Copper EW Intelligent Optimization System — Backend v2.0
=========================================================
Improvements:
  - Real surrogate model loading from local joblib files
  - Real NSGA-II optimization via nsga2_optimization.py
  - Real PPO-Lagrangian via ppo_lagrangian.py
  - Validated CSV/XLSX upload and manual entry with local SQLite persistence
  - WebSocket real-time progress streaming
  - Bounded background tasks with truthful progress and result records
  - Offline cache with SQLite fallback
"""

import os
import sys
import json
import time
import random
import sqlite3
import subprocess
import threading
import traceback
import csv
import io
import math
import uuid
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from process_contract import (MODES, DEFAULTS, BOUNDS, STATE_KEYS, normalize_params,
                              condition_number, finite_number, with_constraints,
                              demo_predict, decode_policy_action, result_row)

DEMO_MODE = os.environ.get('JCP_DEMO', '0') == '1'

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    if DEMO_MODE:
        raise ImportError('Synthetic demo does not load private model artifacts')
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    if DEMO_MODE:
        raise ImportError('Synthetic demo does not require PyTorch')
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
app.config.update(MAX_CONTENT_LENGTH=4 * 1024 * 1024,
                  TRUSTED_HOSTS=['localhost', '127.0.0.1', '[::1]'])
socketio = SocketIO(app, async_mode='threading')

BASE_DIR = Path(__file__).parent
RUNTIME_DIR = Path(os.environ.get('JCP_RUNTIME_DIR', str(BASE_DIR.parents[2] / '.runtime' /
                    ('demo' if DEMO_MODE else 'research')))).expanduser().resolve()
UPLOAD_DIR = RUNTIME_DIR / 'uploads'
CACHE_DIR  = RUNTIME_DIR / 'cache'
OPT_DIR    = RUNTIME_DIR / 'opt_results'
EXPORT_DIR = RUNTIME_DIR / 'exports'
MODEL_DIR = Path(os.environ.get('JCP_MODEL_DIR', str(BASE_DIR.parent / 'modeling' / 'outputs')))
RL_MODEL_DIR = Path(os.environ.get('JCP_POLICY_DIR', str(BASE_DIR / 'pt')))

for d in [UPLOAD_DIR, CACHE_DIR, OPT_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CACHE_DB  = str(CACHE_DIR / 'local_cache.db')


@contextmanager
def db_connection():
    """Commit/rollback and close every connection, including on Windows."""
    connection = sqlite3.connect(CACHE_DB, timeout=15)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


NSGA_SCRIPT = str(BASE_DIR / 'nsga2_optimization.py')
PPO_SCRIPT  = str(BASE_DIR / 'ppo_lagrangian.py')
NSGA_XLSX   = str(BASE_DIR / 'NSGA.xlsx')
RL_XLSX     = str(BASE_DIR / 'RL.xlsx')

# All web-session data uses isolated local SQLite storage.
DB_OK = True

# ── SQLite cache (always available) ─────────────────────────────────────────
def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS process_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, cu_in REAL, temperature REAL,
        current REAL, flow REAL, duration REAL, mode TEXT, source TEXT, synced INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS experiment_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT UNIQUE, algorithm TEXT, hyperparameters TEXT,
        data_version TEXT, runtime REAL, pareto_front TEXT,
        status TEXT DEFAULT 'completed', timestamp TEXT
    )''')
    c.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, content TEXT NOT NULL, timestamp TEXT NOT NULL)')
    conn.commit(); conn.close()

init_cache()

# ── Model cache ──────────────────────────────────────────────────────────────
_models = {}  # keyed by 'three_stage', 'four_stage', 'serial'
_model_lock = threading.Lock()

SURROGATE_PATHS = {
    'three_stage': {
        'cu':      MODEL_DIR / 'cu_three_stage'      / 'extra_trees_three_stage.joblib',
        'as':      MODEL_DIR / 'as_three_stage'      / 'extra_trees_three_stage.joblib',
        'voltage': MODEL_DIR / 'voltage_three_stage' / 'extra_trees_three_stage.joblib',
    },
    'four_stage': {
        'cu':      MODEL_DIR / 'cu_four_stage'       / 'extra_trees_four_stage.joblib',
        'as':      MODEL_DIR / 'as_four_stage'       / 'extra_trees_four_stage.joblib',
        'voltage': MODEL_DIR / 'voltage_four_stage'  / 'extra_trees_four_stage.joblib',
    },
    'serial': {
        'cu':      MODEL_DIR / 'cu_serial'           / 'extra_trees_serial.joblib',
        'as':      MODEL_DIR / 'as_serial'           / 'extra_trees_serial.joblib',
        'voltage': MODEL_DIR / 'voltage_serial'      / 'extra_trees_serial.joblib',
    },
}

# RL Actor model paths
RL_ACTOR_PATHS = {
    1: RL_MODEL_DIR / 'ppo_actor_condition1.pt',
    2: RL_MODEL_DIR / 'ppo_actor_condition2.pt',
    3: RL_MODEL_DIR / 'ppo_actor_condition3.pt',
}

# RL model dimensions per condition
RL_MODEL_dims = {
    1: {'state_dim': 5, 'action_dim': 4, 'weight_dim': 4, 'n_constraints': 5},
    2: {'state_dim': 5, 'action_dim': 4, 'weight_dim': 4, 'n_constraints': 5},
    3: {'state_dim': 8, 'action_dim': 7, 'weight_dim': 4, 'n_constraints': 5},
}

_rl_models = {}  # keyed by condition number (1, 2, 3)
_model_load_finished = DEMO_MODE

class ActorCritic(torch.nn.Module if HAS_TORCH else object):
    def __init__(self, s_dim, a_dim, w_dim=4, hidden=128, n_constraints=5):
        if not HAS_TORCH:
            raise RuntimeError('Local policy inference requires the research PyTorch environment')
        super().__init__()
        inp = s_dim + w_dim
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(inp, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        )
        self.mu = torch.nn.Linear(hidden, a_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(a_dim))
        self.reward_critic = torch.nn.Sequential(
            torch.nn.Linear(inp, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        self.cost_critics = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(inp, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, 1),
            ) for _ in range(n_constraints)
        ])

    def forward(self, s, w):
        x = torch.cat([s, w], dim=-1)
        h = self.shared(x)
        mu = torch.tanh(self.mu(h))
        std = self.log_std.exp().clamp(1e-3, 1.0)
        return mu, std

def load_models_bg():
    """Background model loading — emits socket events on completion."""
    if not HAS_JOBLIB:
        socketio.emit('model_status', {'status': 'unavailable', 'msg': 'joblib not installed'})
        return
    loaded = {}
    for mode, paths in SURROGATE_PATHS.items():
        loaded[mode] = {}
        for target, path in paths.items():
            if path.exists():
                try:
                    m = joblib.load(str(path))
                    if hasattr(m, 'n_jobs'):
                        m.n_jobs = -1
                    loaded[mode][target] = m
                    print(f"[Model] Loaded {mode}/{target}")
                except Exception as e:
                    print(f"[Model] Failed {mode}/{target}: {e}")
            else:
                print(f"[Model] Not found: {path}")
    with _model_lock:
        _models.update(loaded)
    n = sum(len(v) for v in loaded.values())
    print(f"[Model] Total surrogate models loaded: {n}")

    # Load RL actor models
    print(f"[DEBUG] HAS_TORCH = {HAS_TORCH}")
    if HAS_TORCH:
        print(f"[DEBUG] Loading RL models from {RL_MODEL_DIR}")
        for cond, path in RL_ACTOR_PATHS.items():
            print(f"[DEBUG] Checking condition {cond}: {path} (exists={path.exists()})")
            if path.exists():
                try:
                    dims = RL_MODEL_dims.get(cond, {'state_dim': 5, 'action_dim': 4, 'weight_dim': 4})
                    print(f"[DEBUG] Creating ActorCritic with dims: {dims}")
                    model = ActorCritic(
                        s_dim=dims['state_dim'],
                        a_dim=dims['action_dim'],
                        w_dim=dims.get('weight_dim', 4)
                    )
                    print(f"[DEBUG] Loading state dict from {path}")
                    state_dict = torch.load(str(path), map_location='cpu', weights_only=True)
                    model.load_state_dict(state_dict)
                    model.eval()
                    _rl_models[cond] = model
                    print(f"[RL Model] Loaded condition {cond}: {path.name}")
                except Exception as e:
                    print(f"[RL Model] Failed to load condition {cond}: {e}")
            else:
                print(f"[RL Model] Not found: {path}")
    else:
        print("[RL Model] torch not available, skipping RL models")

    n_rl = len(_rl_models)
    socketio.emit('model_status', {'status': 'ready', 'loaded': n, 'rl_loaded': n_rl, 'msg': f'{n} surrogate + {n_rl} RL models loaded'})
    print(f"[Model] Total loaded: {n} surrogate + {n_rl} RL = {n + n_rl} models")

def _load_research_models():
    global _model_load_finished
    try:
        load_models_bg()
    finally:
        _model_load_finished = True


if not DEMO_MODE:
    threading.Thread(target=_load_research_models, daemon=True).start()

# ── Surrogate prediction ──────────────────────────────────────────────────────
def _predict_surrogate(mode: str, params: dict) -> dict:
    condition = next((c for c, name in MODES.items() if name == mode), None)
    if condition is None:
        raise ValueError('Unknown operating mode')
    normalized = normalize_params(condition, params)
    if DEMO_MODE:
        return demo_predict(condition, normalized)
    from model_runtime import predict_local
    with _model_lock:
        available = dict(_models)
    return predict_local(condition, params, available)


# ── Running tasks registry ────────────────────────────────────────────────────
_tasks = {}  # bounded local task registry
_optimization_lock = threading.Lock()

def _new_task_id():
    completed = [k for k, v in _tasks.items() if v['status'] != 'running']
    while len(_tasks) >= 100 and completed:
        _tasks.pop(completed.pop(0), None)
    return 'task_' + uuid.uuid4().hex

# ════════════════════════════════════════════════════════════════════════════════
# DATA MANAGEMENT ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    n_models = sum(len(v) for v in _models.values())
    return jsonify({'status': 'ok', 'mode': 'synthetic_demo' if DEMO_MODE else 'research',
                    'models_loaded': n_models, 'rl_models_loaded': len(_rl_models),
                    'db_ok': True, 'timestamp': datetime.now().isoformat(),
                    'model_load_finished': _model_load_finished,
                    'defaults': DEFAULTS, 'bounds': BOUNDS, 'condition_modes': MODES,
                    'research_training': 'local CLI only',
                    'surrogate_status': 'synthetic_demo' if DEMO_MODE else
                    ('loading' if not _model_load_finished else 'ready' if n_models == 9 else 'unavailable')})

@app.route('/api/data/manual', methods=['POST'])
def data_manual():
    try:
        row = _validate_data_row(request.get_json())
        _sqlite_insert_data(row)
        _log_action('data_import', {'source': 'manual', 'mode': row['mode']})
        return jsonify({'status': 'success', 'record': row, 'message': 'Record validated and saved locally'})
    except (ValueError, TypeError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 422

@app.route('/api/data/upload', methods=['POST'])
def data_upload():
    if not HAS_PANDAS:
        return jsonify({'status': 'error', 'message': 'Install pandas and openpyxl to import files'}), 503
    upload = request.files.get('file')
    if not upload or not upload.filename:
        return jsonify({'status': 'error', 'message': 'Choose a CSV or XLSX file'}), 400
    name = secure_filename(upload.filename)
    extension = Path(name).suffix.lower()
    if extension not in ('.csv', '.xlsx'):
        return jsonify({'status': 'error', 'message': 'Supported types are CSV and XLSX'}), 415
    target = UPLOAD_DIR / (uuid.uuid4().hex + extension)
    try:
        upload.save(target)
        frame = pd.read_csv(target) if extension == '.csv' else pd.read_excel(target)
        if len(frame) > 2000:
            return jsonify({'status': 'error', 'message': 'At most 2,000 rows per upload'}), 422
        aliases = {'Cu_in': 'cu_in', 'T': 'temperature', 'I': 'current', 'Q': 'flow',
                   't': 'duration', 'operation_mode': 'mode'}
        frame = frame.rename(columns=aliases)
        # Validate the whole batch before writing any row.
        rows = [_validate_data_row(r, source='file') for r in frame.to_dict('records')]
        with db_connection() as conn:
            conn.executemany('INSERT INTO process_data '
                '(timestamp,cu_in,temperature,current,flow,duration,mode,source) VALUES (?,?,?,?,?,?,?,?)',
                [(r['timestamp'], r['cu_in'], r['temperature'], r['current'], r['flow'],
                  r['duration'], r['mode'], r['source']) for r in rows])
        _log_action('data_import', {'source': 'file', 'imported': len(rows)})
        return jsonify({'status': 'success', 'imported': len(rows), 'errors': 0,
                        'columns': list(frame.columns), 'message': 'Validated rows saved locally'})
    except (ValueError, TypeError, KeyError):
        return jsonify({'status': 'error', 'message': 'Invalid file or incomplete/out-of-range process rows; no rows were imported'}), 422
    except Exception:
        app.logger.exception('File import failed')
        return jsonify({'status': 'error', 'message': 'Could not read this file; verify its format and required columns'}), 422
    finally:
        target.unlink(missing_ok=True)

@app.route('/api/data/db-import', methods=['POST'])
def data_db_import():
    return jsonify({'status': 'error', 'message':
        'Arbitrary database connections and SQL are disabled in the web interface. '
        'Export approved records to CSV/XLSX and import them locally.'}), 403

@app.route('/api/data/list')
def data_list():
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError('Invalid pagination')
        source_filter = request.args.get('source')
        with db_connection() as conn:
            conn.row_factory = sqlite3.Row
            where, values = (' WHERE source=?', [source_filter]) if source_filter else ('', [])
            total = conn.execute('SELECT COUNT(*) FROM process_data' + where, values).fetchone()[0]
            rows = conn.execute('SELECT * FROM process_data' + where + ' ORDER BY id DESC LIMIT ? OFFSET ?',
                                values + [limit, offset]).fetchall()
        return jsonify({'status': 'success', 'data': [dict(r) for r in rows], 'total': total})
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid pagination', 'data': [], 'total': 0}), 422

@app.route('/api/data/stats')
def data_stats():
    """Data statistics for dashboard."""
    try:
        conn = sqlite3.connect(CACHE_DB)
        total = conn.execute('SELECT COUNT(*) FROM process_data').fetchone()[0]
        by_source = dict(conn.execute(
            'SELECT source, COUNT(*) FROM process_data GROUP BY source').fetchall())
        latest = conn.execute(
            'SELECT timestamp FROM process_data ORDER BY id DESC LIMIT 1').fetchone()
        conn.close()
        return jsonify({
            'total': total,
            'by_source': by_source,
            'latest': latest[0] if latest else None,
        })
    except:
        return jsonify({'total': 0, 'by_source': {}, 'latest': None})

# ════════════════════════════════════════════════════════════════════════════════
# SURROGATE PREDICTION
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/surrogate/predict', methods=['POST'])
def surrogate_predict():
    try:
        d = request.get_json()
        if not isinstance(d, dict):
            raise ValueError('A JSON object is required')
        condition = condition_number(d.get('condition', 1))
        params = normalize_params(condition, d.get('params', {}))
        if 'timestamp' in d.get('params', {}):
            params['timestamp'] = d['params']['timestamp']
        result = _predict_surrogate(MODES[condition], params)
        _log_action('surrogate_predict', {'condition': condition, 'source': result['model_source']})
        return jsonify({'status': 'success', 'result': result})
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 422
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 503
    except Exception:
        app.logger.exception('Surrogate inference failed')
        return jsonify({'status': 'error', 'message': 'Local model inference failed; no substitute result was generated'}), 503

# ════════════════════════════════════════════════════════════════════════════════
# NSGA-II OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/nsga2/run', methods=['POST'])
def nsga2_run():
    if not DEMO_MODE:
        return jsonify({'status': 'error', 'message':
            'Research optimization is run through the audited local CLI. '
            'Use run_demo.py for the synthetic NSGA-II workflow.'}), 409
    try:
        data = request.get_json()
        params = data.get('params', {})
        if params.get('run_mode', 'online') not in ('online', 'manual', 'excel'):
            raise ValueError('Unknown optimization input mode')
        condition = condition_number(params.get('condition', 1))
        algorithm = params.get('algorithm_params', {})
        population = _bounded_int(algorithm.get('pop', 64), 20, 256, 'population')
        generations = _bounded_int(algorithm.get('gen', 30), 5, 200, 'generations')
        seed = _bounded_int(algorithm.get('seed', 42), 0, 2147483647, 'seed')
        n_solutions = _bounded_int(algorithm.get('n_solutions', 50), 4, 100, 'retained solutions')
        crossover = finite_number(algorithm.get('pc', 0.9), 'crossover probability')
        eta = finite_number(algorithm.get('eta', 15), 'crossover eta')
        if not 0 < crossover <= 1 or not 1 <= eta <= 100:
            raise ValueError('Invalid crossover settings')
        if params.get('run_mode') == 'excel':
            raise ValueError('Import the file in Data Management first, then use the latest-record input mode')
        inputs = dict(DEFAULTS[condition])
        if params.get('run_mode') == 'manual':
            manual = params.get('manual_data', {})
            inputs['Cu_in'] = manual.get('cu_in', inputs['Cu_in'])
            for unit in ('A', 'B'):
                if f'T_{unit}' in inputs:
                    inputs[f'T_{unit}'] = manual.get('temperature', inputs[f'T_{unit}'])
        else:
            with db_connection() as conn:
                conn.row_factory = sqlite3.Row
                latest = conn.execute('SELECT * FROM process_data WHERE mode=? ORDER BY id DESC LIMIT 1',
                                      (MODES[condition],)).fetchone()
            if latest:
                inputs = _row_params(dict(latest), condition)
        inputs = normalize_params(condition, inputs)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 422
    if not _optimization_lock.acquire(blocking=False):
        return jsonify({'status': 'error', 'message': 'An optimization is already running; wait for it to finish'}), 409
    task_id = _new_task_id()
    _tasks[task_id] = {'status': 'running', 'progress': 5, 'result': None}
    def run():
        start = time.perf_counter()
        try:
            from demo_optimization import run_demo_nsga2
            result = run_demo_nsga2(condition, inputs, population=population, generations=generations,
                                    seed=seed, crossover=crossover, eta=eta, n_solutions=n_solutions)
            record_params = {'condition': condition, 'run_mode': params.get('run_mode', 'online'),
                             'algorithm_params': {'population': population, 'generations': generations,
                                                  'seed': seed, 'crossover': crossover, 'eta': eta},
                             'inputs': inputs, 'model_source': 'synthetic_demo',
                             'actual_evaluations': result['evaluations']}
            _save_experiment(task_id, 'NSGA-II (synthetic demo)', record_params,
                             result, runtime=time.perf_counter() - start)
            _tasks[task_id].update(status='completed', progress=100, result=result)
            socketio.emit('nsga2_complete', {'task_id': task_id, 'success': True, **result})
        except Exception:
            app.logger.exception('NSGA-II task failed')
            message = 'Optimization failed; no substitute results were generated. Check the local server log.'
            _tasks[task_id].update(status='failed', progress=100, error=message)
            socketio.emit('nsga2_error', {'task_id': task_id, 'error': message})
        finally:
            _optimization_lock.release()
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status': 'success', 'task_id': task_id})



# ════════════════════════════════════════════════════════════════════════════════
# PPO-Lagrangian RL
# ════════════════════════════════════════════════════════════════════════════════

@app.route('/api/rl/run', methods=['POST'])
def rl_run():
    return jsonify({'status': 'error', 'message':
        'PPO training requires private research assets and is an explicit local CLI operation. '
        'The web interface never simulates completed training. Local saved-policy inference is available in research mode.'}), 409



@app.route('/api/rl/predict', methods=['POST'])
def rl_predict():
    if DEMO_MODE:
        return jsonify({'status': 'error', 'message': 'Synthetic demo does not pretend to load a trained PPO policy'}), 409
    try:
        data = request.get_json()
        condition = condition_number(data.get('condition', 1))
        params = normalize_params(condition, data.get('params', {}))
        if not HAS_TORCH or condition not in _rl_models:
            raise RuntimeError('The local policy is unavailable; configure JCP_POLICY_DIR')
        weights = data.get('weights', [0.25] * 4)
        if not isinstance(weights, list) or len(weights) != 4:
            raise ValueError('Provide four non-negative preference weights')
        weights = [finite_number(w, 'weight') for w in weights]
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError('Preference weights must be non-negative with a positive sum')
        weights = [w / sum(weights) for w in weights]
        state = [params[key] for key in STATE_KEYS[condition]]
        with torch.no_grad():
            action, _ = _rl_models[condition](torch.tensor([state], dtype=torch.float32),
                                             torch.tensor([weights], dtype=torch.float32))
        updated = decode_policy_action(condition, params, action[0].cpu().tolist(), fixed_feed=True)
        if 'timestamp' in data.get('params', {}):
            updated['timestamp'] = data['params']['timestamp']
        prediction = _predict_surrogate(MODES[condition], updated)
        result = result_row(condition, updated, prediction, scene='Policy inference')
        unit = 'B' if condition == 2 else 'A'
        result.update(I_opt=updated[f'I_{unit}'], T_opt=updated[f'T_{unit}'],
                      Q_opt=updated[f'Q_{unit}'], updated_params=updated, weights=weights)
        return jsonify({'status': 'success', 'model_source': 'real', 'result': result})
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 422
    except RuntimeError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 503
    except Exception:
        app.logger.exception('Policy inference failed')
        return jsonify({'status': 'error', 'message': 'Policy evaluation failed; no fallback recommendation was generated'}), 503


@app.route('/api/task/<task_id>')
def task_status(task_id):
    t = _tasks.get(task_id, {'status': 'not_found'})
    return jsonify({k: v for k, v in t.items() if k != 'process'}), (404 if t['status'] == 'not_found' else 200)

# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RECORDS
# ════════════════════════════════════════════════════════════════════════════════

def _save_experiment(task_id, algorithm, params, result, runtime):
    with db_connection() as conn:
        conn.execute('INSERT INTO experiment_records '
            '(experiment_id,algorithm,hyperparameters,data_version,runtime,pareto_front,status,timestamp) '
            'VALUES (?,?,?,?,?,?,?,?)', (task_id, algorithm, json.dumps(params, allow_nan=False),
            'synthetic-demo-v1' if DEMO_MODE else 'local-research', float(runtime),
            json.dumps(result, allow_nan=False), 'completed', datetime.now().isoformat()))

@app.route('/api/experiments')
def experiments():
    with db_connection() as conn:
        conn.row_factory = sqlite3.Row
        records = conn.execute('SELECT * FROM experiment_records ORDER BY id DESC LIMIT 50').fetchall()
    return jsonify([_decode_record(dict(r)) for r in records])


@app.route('/api/experiments/<exp_id>')
def experiment_detail(exp_id):
    with db_connection() as conn:
        conn.row_factory = sqlite3.Row
        record = conn.execute('SELECT * FROM experiment_records WHERE experiment_id=?', (exp_id,)).fetchone()
    if record is None:
        return jsonify({'status': 'not_found', 'message': 'Experiment not found'}), 404
    return jsonify(_decode_record(dict(record)))

@app.route('/api/export', methods=['POST'])
def export_results():
    data = request.get_json(silent=True) or {}
    if data.get('format', 'csv') != 'csv':
        return jsonify({'status': 'error', 'message': 'Only CSV export is supported'}), 422
    with db_connection() as conn:
        record = conn.execute('SELECT pareto_front FROM experiment_records WHERE experiment_id=?',
                              (str(data.get('experiment_id', '')),)).fetchone()
    if record is None:
        return jsonify({'status': 'not_found', 'message': 'Experiment not found'}), 404
    result = json.loads(record[0])
    rows = result.get('front', [])
    if not rows:
        return jsonify({'status': 'error', 'message': 'This experiment has no candidate rows'}), 409
    columns = ['condition','Cu_in','T_A','I_A','Q_A','T_B','I_B','Q_B','t',
               'Cu_out','As_out','E','profit','feasible','model_source']
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
    name = 'experiment_' + uuid.uuid4().hex + '.csv'
    (EXPORT_DIR / name).write_text(buffer.getvalue(), encoding='utf-8-sig')
    return jsonify({'status': 'success', 'filename': name, 'url': '/exports/' + name,
                    'rows': len(rows), 'message': 'Actual computed candidate rows exported'})

# ════════════════════════════════════════════════════════════════════════════════
# ADMIN / LOGS
# ════════════════════════════════════════════════════════════════════════════════

def _sqlite_insert_data(row):
    conn = sqlite3.connect(CACHE_DB)
    conn.execute('''INSERT INTO process_data (timestamp, cu_in, temperature, current, flow, duration, mode, source)
                    VALUES (?,?,?,?,?,?,?,?)''',
                 (row['timestamp'], row['cu_in'], row['temperature'], row['current'],
                  row['flow'], row['duration'], row['mode'], row.get('source','manual')))
    conn.commit(); conn.close()

_logs = []
def _log_action(action, details=None):
    entry = {'timestamp': datetime.now().isoformat(), 'action': action, 'details': details or {}}
    _logs.append(entry)
    if len(_logs) > 500:
        _logs.pop(0)

@app.route('/api/logs')
def get_logs():
    return jsonify(list(reversed(_logs[-100:])))

@app.route('/api/users')
def get_users():
    # This is a local research tool, not a multi-user authentication service.
    return jsonify([])

@app.route('/api/feedback')
def get_feedback():
    with db_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM feedback ORDER BY id DESC LIMIT 50').fetchall()
    return jsonify([dict(row) for row in rows])

@app.route('/api/feedback', methods=['POST'])
def add_feedback():
    data = request.get_json(silent=True) or {}
    content = data.get('content', data.get('feedback', ''))
    if not isinstance(content, str) or not 1 <= len(content.strip()) <= 2000:
        return jsonify({'status': 'error', 'message': 'Feedback must contain 1 to 2,000 characters'}), 422
    with db_connection() as conn:
        conn.execute('INSERT INTO feedback (content,timestamp) VALUES (?,?)',
                     (content.strip(), datetime.now().isoformat()))
    return jsonify({'status': 'success'})

# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('connected', {'msg': 'CuEW local backend connected'})
    on_ping_models()

@socketio.on('ping_models')
def on_ping_models():
    count = sum(len(v) for v in _models.values())
    emit('model_status', {'status': 'demo' if DEMO_MODE else ('ready' if count else 'unavailable'),
                         'loaded': count, 'rl_loaded': len(_rl_models), 'demo': DEMO_MODE})

# ── Static files ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'copper_ew_gui_en.html')

@app.route('/<path:path>')
def static_files(path):
    # Never expose source, uploaded data, model weights, database or runtime files.
    if path == 'release_workflow.js':
        return send_from_directory(BASE_DIR, path)
    if path == 'favicon.ico':
        return '', 204
    abort(404)


# Shared input/storage helpers and local-only HTTP boundary.
def _bounded_int(value, low, high, label):
    number = finite_number(value, label)
    if number != int(number) or not low <= number <= high:
        raise ValueError(f'{label} must be an integer between {low} and {high}')
    return int(number)


def _validate_data_row(data, source='manual'):
    if not isinstance(data, dict):
        raise ValueError('A record must be an object')
    mode = data.get('mode')
    if mode not in MODES.values():
        raise ValueError('mode must be three_stage, four_stage, or serial')
    condition = next(c for c, m in MODES.items() if m == mode)
    required = ('cu_in', 'temperature', 'current', 'flow', 'duration')
    if any(k not in data for k in required):
        raise ValueError('Required columns: cu_in, temperature, current, flow, duration, mode')
    row = {k: finite_number(data[k], k) for k in required}
    # A single-row import records one common setpoint for both series units.
    normalize_params(condition, _row_params({**row, 'mode': mode}, condition))
    return {**row, 'mode': mode, 'source': source, 'timestamp': datetime.now().isoformat()}


def _row_params(row, condition):
    params = {'Cu_in': row['cu_in'], 't': row['duration']}
    for unit in (('A','B') if condition == 3 else (('A',) if condition == 1 else ('B',))):
        params.update({f'T_{unit}': row['temperature'], f'I_{unit}': row['current'], f'Q_{unit}': row['flow']})
    return params


def _decode_record(record):
    for key in ('hyperparameters', 'pareto_front'):
        record[key] = json.loads(record[key]) if record.get(key) else {}
    return record


@app.before_request
def _same_origin_writes():
    if request.method in ('POST','PUT','PATCH','DELETE'):
        origin = request.headers.get('Origin')
        if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
            return jsonify({'status':'error','message':'Cross-origin writes are not allowed'}), 403


@app.after_request
def _security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/exports/<filename>')
def download_export(filename):
    if not filename.startswith('experiment_') or not filename.endswith('.csv') or secure_filename(filename) != filename:
        abort(404)
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True, mimetype='text/csv')


@app.route('/api/data/preprocess', methods=['POST'])
def preprocess_records():
    # Imports are already validated; this endpoint reports an actual read-only audit.
    with db_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM process_data').fetchall()
    invalid = sum(1 for row in rows if not _record_is_valid(dict(row)))
    return jsonify({'status':'success','checked':len(rows),'invalid':invalid,
                    'message':'Validation audit completed; no records were modified'})


def _record_is_valid(row):
    try:
        _validate_data_row(row)
        return True
    except (ValueError,TypeError):
        return False


if __name__ == '__main__':
    print("=" * 60)
    print(" Copper EW Intelligent Optimization System — Backend v2.0")
    print(f" Base dir: {BASE_DIR}")
    print(f" Models dir: {MODEL_DIR}")
    print(" Database: local runtime storage; sensitive connection values are not displayed")
    print("=" * 60)
    socketio.run(app, debug=False, host='127.0.0.1', port=int(os.environ.get('JCP_PORT', '5001')), use_reloader=False, allow_unsafe_werkzeug=True)
