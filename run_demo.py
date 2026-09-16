"""Start the public, synthetic-data demonstration on loopback only."""
import os
from pathlib import Path
import sys

os.environ['JCP_DEMO'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src/esrlcmo_project/interface'))
from app_v2 import app, socketio

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=int(os.environ.get('JCP_PORT', '5001')),
                 debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
