"""Empirical installed-build check, not part of the application launcher."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EXE = '/home/vimal/AI/llama.cpp/build/bin/llama-server'
MODEL = ROOT / 'tests/models/SmolLM2-135M-Instruct-Q4_K_M.gguf'
OUT = ROOT / 'verification'
OUT.mkdir(exist_ok=True)
argv = [EXE, '--model', str(MODEL), '--host', '127.0.0.1', '--port', '18096',
        '--ctx-size', '1024', '--batch-size', '128', '--ubatch-size', '64',
        '--parallel', '1', '--device', 'CUDA0,Vulkan1', '--gpu-layers', '0',
        '--override-tensor', r'^blk\.[0-9]\..*=CUDA0,^blk\.1[0-9]\..*=Vulkan1',
        '--fit', 'off', '--no-kv-offload', '--load-mode', 'mmap', '--metrics', '--verbosity', '4']
env = {k:v for k,v in os.environ.items() if not k.startswith('LLAMA_')}
with (OUT/'mixed-backend.log').open('w') as log:
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, proc.pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    def send_signal(sig):
        if libc.syscall(424, fd, sig, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')
    try:
        deadline = time.monotonic()+90
        healthy = False
        while time.monotonic()<deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen('http://127.0.0.1:18096/health',timeout=2) as r:
                    healthy = r.status == 200
                if healthy: break
            except Exception:
                time.sleep(.25)
        if not healthy:
            raise RuntimeError(f'No healthy server; returncode={proc.poll()}')
        req = urllib.request.Request('http://127.0.0.1:18096/completion',
              json.dumps({'prompt':'The capital of France is','n_predict':12,'temperature':0,'seed':42}).encode(),
              {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=45) as r: result = json.load(r)
        report = {'argv':argv,'health':healthy,'content':result.get('content'),'timings':result.get('timings')}
        (OUT/'mixed-backend.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:
        if proc.poll() is None:
            send_signal(signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                send_signal(signal.SIGKILL)
                proc.wait(timeout=5)
        os.close(fd)
        print('Process stopped:', proc.returncode)
