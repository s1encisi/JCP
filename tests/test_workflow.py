"""Public tests use generated inputs and a temporary SQLite database only."""
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

INTERFACE = Path(__file__).resolve().parents[1] / 'src/esrlcmo_project/interface'
sys.path.insert(0, str(INTERFACE))
from process_contract import (DEFAULTS, BOUNDS, CATHODE_AREA, normalize_params,
                              demo_predict, check_constraints, decode_policy_action)


class ProcessContractTests(unittest.TestCase):
    def test_three_conditions_are_deterministic_and_finite(self):
        for condition in (1, 2, 3):
            result = demo_predict(condition, DEFAULTS[condition])
            self.assertEqual(result, demo_predict(condition, DEFAULTS[condition]))
            self.assertEqual(result['model_source'], 'synthetic_demo')
            self.assertEqual(len(result['constraints']), 5)
            json.dumps(result, allow_nan=False)

    def test_rejects_nan_infinity_boolean_and_missing_values(self):
        for bad in (float('nan'), float('inf'), True, 'not-a-number'):
            with self.assertRaises(ValueError):
                normalize_params(1, {**DEFAULTS[1], 'I_A': bad})
        with self.assertRaises(ValueError): normalize_params(1, {})
        with self.assertRaises(ValueError): normalize_params(4, DEFAULTS[1])

    def test_aliases_preserve_units(self):
        self.assertEqual(normalize_params(1, {'Cu_in': 38, 'T': 58, 'I': 18000, 'Q': 117, 't': 4}), DEFAULTS[1])

    def test_all_constraint_directions_match_the_process_contract(self):
        p = {**DEFAULTS[1], 'I_A': 40000}
        constraints = check_constraints(1, p, {'Cu_out': 9, 'As_out': 3, 'V': 48})
        self.assertTrue(all(not value['ok'] for value in constraints.values()))
        self.assertEqual(constraints['C2_As']['operator'], '>=')
        self.assertEqual(constraints['C5_CuAs']['operator'], '<=')
        self.assertAlmostEqual(constraints['C3_J']['val'], 40000 / CATHODE_AREA)

    def test_constraint_boundaries_and_zero_arsenic(self):
        p = {**DEFAULTS[1], 'I_A': 340*CATHODE_AREA}
        r = check_constraints(1, p, {'Cu_out': 3.2, 'As_out': 4, 'V': 40})
        self.assertTrue(all(v['ok'] for v in r.values()))
        r = check_constraints(1, p, {'Cu_out': 3.2, 'As_out': 0, 'V': 40})
        self.assertIsNone(r['C5_CuAs']['val'])
        self.assertFalse(r['C5_CuAs']['ok'])

    def test_policy_actions_are_increments_with_fixed_feed(self):
        result = decode_policy_action(1, DEFAULTS[1], [1, .5, .5, -.5])
        self.assertEqual(result['Cu_in'], 38)
        self.assertEqual(result['T_A'], 59)
        self.assertEqual(result['I_A'], 18500)
        self.assertEqual(result['Q_A'], 116.5)
        self.assertEqual(result['t'], 4)

    def test_serial_action_updates_both_units_and_checks_length(self):
        result = decode_policy_action(3, DEFAULTS[3], [1, .5, .5, -.5, -.5, -.5, .5])
        self.assertEqual(result['T_B'], 58)
        self.assertEqual(result['I_B'], 15500)
        self.assertEqual(result['Q_B'], 118.5)
        with self.assertRaises(ValueError): decode_policy_action(3, DEFAULTS[3], [0]*4)
        with self.assertRaises(ValueError): decode_policy_action(1, DEFAULTS[1], [0,0,2,0])

    def test_missing_research_assets_fail_without_fallback(self):
        from model_runtime import predict_local
        with self.assertRaises(RuntimeError): predict_local(1, DEFAULTS[1], {})


class WorkflowApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tempfile.TemporaryDirectory(prefix='jcp-workflow-test-')
        cls.env = patch.dict(os.environ, {'JCP_DEMO': '1', 'JCP_RUNTIME_DIR': cls.runtime.name})
        cls.env.start()
        spec = importlib.util.spec_from_file_location('tested_jcp_backend', INTERFACE / 'app_v2.py')
        cls.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.backend)
        cls.backend.app.config['TESTING'] = True

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.runtime.cleanup()

    def setUp(self):
        self.client = self.backend.app.test_client()
        self.assertTrue(self.backend.CACHE_DB.startswith(self.runtime.name))
        with self.backend.db_connection() as conn:
            for table in ('process_data', 'experiment_records', 'feedback'):
                conn.execute('DELETE FROM ' + table)

    def test_demo_boots_without_private_models_or_pytorch(self):
        r = self.client.get('/api/health').get_json()
        self.assertEqual(r['mode'], 'synthetic_demo')
        self.assertFalse(self.backend.HAS_TORCH)
        self.assertEqual(r['models_loaded'], 0)
        with self.client.get('/') as response:
            self.assertIn('runtime-notice', response.get_data(as_text=True))
        with self.client.get('/release_workflow.js') as response:
            self.assertEqual(response.status_code, 200)

    def test_rejects_cross_origin_writes_and_untrusted_hosts(self):
        r = self.client.post('/api/surrogate/predict', json={}, headers={'Origin': 'https://untrusted.example'})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.client.get('/api/health', headers={'Host': 'untrusted.example'}).status_code, 400)

    def test_private_file_urls_are_not_served(self):
        for path in ('app_v2.py','copper_ew.db','pt/ppo_actor_condition1.pt','uploads/data.xlsx',
                     '.git/config','process_contract.py','../README.md'):
            self.assertEqual(self.client.get('/'+path).status_code, 404, path)

    def test_prediction_checks_five_constraints_and_input_errors(self):
        for condition in (1, 2, 3):
            r = self.client.post('/api/surrogate/predict', json={'condition': condition, 'params': DEFAULTS[condition]})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(len(r.get_json()['result']['constraints']), 5)
        r = self.client.post('/api/surrogate/predict', json={'condition': 1, 'params': {'Cu_in': 38}})
        self.assertEqual(r.status_code, 422)

    @staticmethod
    def sample_row():
        return {'cu_in': 38, 'temperature': 58, 'current': 18000, 'flow': 117,
                'duration': 4, 'mode': 'three_stage'}

    def test_manual_import_pagination_and_actual_validation_audit(self):
        self.assertEqual(self.client.post('/api/data/manual', json=self.sample_row()).status_code, 200)
        r = self.client.get('/api/data/list?source=manual').get_json()
        self.assertEqual(r['total'], 1)
        self.assertEqual(r['data'][0]['cu_in'], 38)
        self.assertEqual(self.client.get('/api/data/list?source=file').get_json()['total'], 0)
        self.assertEqual(self.client.get('/api/data/list?limit=-1').status_code, 422)
        audit = self.client.post('/api/data/preprocess', json={}).get_json()
        self.assertEqual((audit['checked'], audit['invalid']), (1, 0))

    def test_file_validation_is_atomic_and_upload_names_do_not_escape(self):
        invalid = b'cu_in,temperature,current,flow,duration,mode\n38,58,18000,117,4,three_stage\nNaN,58,18000,117,4,three_stage\n'
        r = self.client.post('/api/data/upload', data={'file': (io.BytesIO(invalid), '../../escape.csv')})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.client.get('/api/data/stats').get_json()['total'], 0)
        valid = invalid.split(b'NaN')[0]
        r = self.client.post('/api/data/upload', data={'file': (io.BytesIO(valid), '../../escape.csv')})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()['imported'], 1)
        self.assertEqual(list(self.backend.UPLOAD_DIR.iterdir()), [])

    def test_research_and_external_database_operations_are_explicit(self):
        self.assertEqual(self.client.post('/api/rl/run', json={}).status_code, 409)
        self.assertEqual(self.client.post('/api/rl/predict', json={}).status_code, 409)
        self.assertEqual(self.client.post('/api/data/db-import', json={}).status_code, 403)
        self.assertEqual(self.client.get('/api/experiments').get_json(), [])
        self.assertEqual(self.client.get('/api/experiments/missing').status_code, 404)
        self.assertEqual(self.client.post('/api/export', json={'experiment_id': 'missing'}).status_code, 404)

    def test_feedback_is_persisted_and_validated(self):
        self.assertEqual(self.client.post('/api/feedback', json={'content':'Synthetic test feedback'}).status_code, 200)
        self.assertEqual(self.client.get('/api/feedback').get_json()[0]['content'], 'Synthetic test feedback')
        self.assertEqual(self.client.post('/api/feedback', json={'content':''}).status_code, 422)

    def test_search_budget_and_concurrent_submission_are_bounded(self):
        self.assertEqual(self.client.post('/api/nsga2/run', json={'params': {'algorithm_params': {'pop': 100000}}}).status_code, 422)
        with self.backend._optimization_lock:
            self.assertEqual(self.client.post('/api/nsga2/run', json={'params': {}}).status_code, 409)

    def test_optimization_history_and_export_use_the_same_computed_rows(self):
        started = self.client.post('/api/nsga2/run', json={'params': {'condition': 1, 'run_mode': 'online',
                  'algorithm_params': {'pop': 24, 'gen': 8, 'seed': 7, 'n_solutions': 12}}})
        self.assertEqual(started.status_code, 200)
        task_id = started.get_json()['task_id']
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            task = self.client.get('/api/task/'+task_id).get_json()
            if task['status'] != 'running': break
            time.sleep(.05)
        self.assertEqual(task['status'], 'completed', task)
        result = task['result']
        self.assertGreater(result['evaluations'], 0)
        self.assertTrue(all(row['feasible'] for row in result['front']))
        self.assertTrue(all(row['Cu_in']==38 and row['t']==4 for row in result['front']))
        self.assertEqual(len(result['results']['cond1']), 4)
        record = self.client.get('/api/experiments/'+task_id).get_json()
        self.assertGreater(record['runtime'], 0)
        self.assertEqual(record['pareto_front']['front'], result['front'])
        exported = self.client.post('/api/export', json={'experiment_id':task_id, 'format':'csv'}).get_json()
        download = self.client.get(exported['url'])
        self.addCleanup(download.close)
        self.assertEqual(download.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(download.get_data().decode('utf-8-sig'))))
        self.assertEqual(len(rows), len(result['front']))
        self.assertAlmostEqual(float(rows[0]['Cu_out']), result['front'][0]['Cu_out'])


if __name__ == '__main__':
    unittest.main()
