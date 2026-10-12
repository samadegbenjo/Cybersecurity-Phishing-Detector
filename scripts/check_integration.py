"""Real HTTP loopback + shared popup JS client; no native Chrome claims."""
import os
from pathlib import Path
import subprocess
import socket
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,'scripts/export_record.py'],cwd=ROOT,check=True)
with socket.socket() as probe:
    probe.bind(('127.0.0.1',5000))  # Fail before accidentally testing another running API.
process=subprocess.Popen([sys.executable,'app.py'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        if process.poll() is not None: raise RuntimeError('API failed to start (port already in use?)')
        try:
            with urlopen('http://127.0.0.1:5000/health',timeout=.5) as r:
                import json
                health=json.load(r)
                if not health['model_ready'] or not health['live_model_ready']: raise RuntimeError('Train both offline and live models first')
            break
        except OSError: time.sleep(.1)
    else: raise RuntimeError('API did not become ready')
    env=dict(os.environ,PHISHING_INTEGRATION='1')
    subprocess.run(['node','--test','tests/extension.test.cjs'],cwd=ROOT,env=env,check=True)
finally:
    process.terminate()
    process.wait(timeout=5)
subprocess.run(['node','-e',"const c=require('./extension/client.js'); c.evaluateRecord({}).then(r=>{if(r.state!=='unable')process.exit(1); console.log('Actual unreachable backend:',r.text)})"],cwd=ROOT,check=True)

subprocess.run(["node","-e","const c=require('./extension/client.js'); c.evaluatePage('http://127.0.0.1:8000/controlled.html').then(r=>{if(r.state!=='unable')process.exit(1); console.log('Actual unreachable backend for active page:',r.text)})"],cwd=ROOT,check=True)
