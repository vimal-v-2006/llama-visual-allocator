"""Real OS lifecycle tests using explicit non-inference fixtures."""
import os
from pathlib import Path
import tempfile
import time
import unittest


def wait_for(predicate, timeout=8):
    end = time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Timed out waiting for fixture')


class RuntimeTests(unittest.TestCase):
    def test_detached_descendant_is_reaped_after_leader_exit(self):
        from allocator.runtime import Runtime
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp)/'fixture'
            marker = Path(temp)/'child'
            fixture.write_text('#!/usr/bin/python3\nimport os,time\np=os.fork()\nif p == 0:\n os.setsid()\n open('+repr(str(marker))+',"w").write(str(os.getpid()))\n time.sleep(60)\nelse:\n time.sleep(.2)\n')
            fixture.chmod(0o700)
            runtime = Runtime()
            try:
                runtime.start([str(fixture)], {'host':'127.0.0.1','port':18997})
                wait_for(marker.exists)
                child = int(marker.read_text())
                wait_for(lambda: not runtime.snapshot()['running'])
                self.assertFalse(Path(f'/proc/{child}').exists())
                runtime.start([str(fixture)], {'host':'127.0.0.1','port':18997})
                runtime.stop()
            finally:
                runtime.stop()

    def test_launch_environment_network_and_busy_port(self):
        from allocator.runtime import Runtime, clean_environment
        from unittest.mock import patch
        import socket
        self.assertEqual(clean_environment({'GGML_CUDA_DISABLE_GRAPHS':'1'})['GGML_CUDA_DISABLE_GRAPHS'],'1')
        for env in ({'LD_PRELOAD':'evil'}, {'GGML_CUDA_DISABLE_GRAPHS':'0'}, {'PATH':'evil'}):
            with self.assertRaises(ValueError): clean_environment(env)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); sock.listen()
            server={'host':'0.0.0.0','port':sock.getsockname()[1],'allow_network':True}
            with patch('allocator.runtime.subprocess.Popen') as spawn:
                with self.assertRaises((ValueError,OSError)): Runtime().start(['/usr/bin/true'],server)
                spawn.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            exe=Path(temp)/'fixture'
            exe.write_text('#!/usr/bin/python3\nimport os,time\nprint("graphs="+str(os.getenv("GGML_CUDA_DISABLE_GRAPHS")),flush=True)\ntime.sleep(60)\n')
            exe.chmod(0o700)
            runtime=Runtime()
            try:
                runtime.start([str(exe)],server,env={'GGML_CUDA_DISABLE_GRAPHS':'1'})
                wait_for(lambda:any('graphs=1' in x for x in runtime.snapshot()['logs']))
            finally: runtime.stop()

    def test_health_does_not_trust_foreign_listener_after_preflight(self):
        from allocator.runtime import Runtime
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading, socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Path(tmp)/'fixture'; fixture.write_text('#!/usr/bin/python3\nimport time\ntime.sleep(60)\n'); fixture.chmod(0o700)
            runtime=Runtime()
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self): self.send_response(200); self.end_headers()
                def log_message(self,*args): pass
            try:
                runtime.start([str(fixture)],{'host':'127.0.0.1','port':port})
                with HTTPServer(('127.0.0.1',port), Handler) as foreign:
                    thread=threading.Thread(target=foreign.serve_forever,daemon=True); thread.start()
                    try: self.assertNotEqual(runtime.snapshot()['health'],'healthy')
                    finally: foreign.shutdown(); thread.join()
            finally: runtime.stop()

    def test_stable_handles_preflight_before_spawn(self):
        from allocator.runtime import Runtime
        from unittest.mock import patch
        with patch('allocator.runtime.pidfd_open',side_effect=OSError('unsupported')), patch('allocator.runtime.subprocess.Popen') as spawn:
            with self.assertRaises(OSError): Runtime().start(['/usr/bin/true'],{'host':'127.0.0.1','port':18997})
            spawn.assert_not_called()

    def test_pidfd_acquisition_failure_cannot_spawn_server(self):
        from allocator.runtime import Runtime
        from allocator.supervisor import pidfd_open
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'spawned'; fixture=Path(tmp)/'fixture'
            fixture.write_text('#!/usr/bin/python3\nopen('+repr(str(marker))+',"w").write("bad")\n'); fixture.chmod(0o700)
            count=0
            def acquire(pid):
                nonlocal count
                count += 1
                if count == 1: return pidfd_open(pid)
                time.sleep(.15)  # Supervisor gets time to run, but must await grant.
                raise OSError('acquisition failure')
            runtime=Runtime()
            with patch('allocator.runtime.pidfd_open',side_effect=acquire):
                with self.assertRaises(OSError): runtime.start([str(fixture)],{'host':'127.0.0.1','port':18997})
            self.assertFalse(marker.exists())
            self.assertIsNone(runtime.proc)
            self.assertFalse(runtime.snapshot()['running'])

    def test_private_user_group_executable_is_accepted(self):
        from allocator.runtime import validate_executable
        from unittest.mock import patch
        import grp, pwd
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'server'; exe.write_text('#!/bin/true\n'); exe.chmod(0o775)
            user=pwd.getpwuid(os.getuid())
            private=grp.struct_group((user.pw_name,'x',os.stat(exe).st_gid,[]))
            with patch('grp.getgrgid',return_value=private), patch('pwd.getpwall',return_value=[user]):
                self.assertEqual(validate_executable(str(exe)),str(exe))
            shared=grp.struct_group(('shared','x',os.stat(exe).st_gid,[user.pw_name,'untrusted']))
            with patch('grp.getgrgid',return_value=shared), patch('pwd.getpwall',return_value=[user]):
                with self.assertRaises(ValueError): validate_executable(str(exe))
            exe.chmod(0o777)
            with self.assertRaises(ValueError): validate_executable(str(exe))

    def test_executable_validation(self):
        from allocator.runtime import validate_executable
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fixture'
            path.write_text('#!/bin/sh\nexit 0\n')
            with self.assertRaises(ValueError):
                validate_executable(str(path))
            path.chmod(0o777)
            with self.assertRaises(ValueError):
                validate_executable(str(path))
            path.chmod(0o700)
            link = Path(temp)/'link'
            link.symlink_to(path)
            with self.assertRaises((ValueError,OSError)):
                validate_executable(str(link))
            self.assertEqual(validate_executable(str(path)), str(path))

    def test_health_reads_real_loopback_fixture(self):
        from allocator.runtime import Runtime
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp)/'fixture'
            fixture.write_text('#!/usr/bin/python3\nfrom http.server import HTTPServer,BaseHTTPRequestHandler\nclass H(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200);self.end_headers();self.wfile.write(b"{\\"status\\":\\"ok\\"}")\nHTTPServer(("127.0.0.1",18997),H).serve_forever()\n')
            fixture.chmod(0o700)
            runtime = Runtime()
            try:
                runtime.start([str(fixture)], {'host':'127.0.0.1','port':18997})
                wait_for(lambda: runtime.snapshot()['health'] == 'healthy')
            finally:
                runtime.stop()

    def test_gpu_sampling_is_cached_and_drm_clients_deduplicated(self):
        from allocator.runtime import Runtime, parse_drm_telemetry
        from unittest.mock import patch
        text='drm-driver: amdgpu\ndrm-pdev: 0000:03:00.0\ndrm-client-id: 7\ndrm-memory-vram: 12 KiB\n'
        self.assertEqual(parse_drm_telemetry([text,text]),{'0000:03:00.0':12288})
        runtime=Runtime()
        sample={'gpu_process_bytes':{},'gpu_global':[{'id':'fixture'}],'gpu_telemetry_warnings':[]}
        with patch('allocator.runtime.sample_gpu_telemetry',return_value=sample) as sample_gpu:
            self.assertEqual(runtime.snapshot()['gpu_global'],sample['gpu_global'])
            runtime.snapshot()
            self.assertEqual(sample_gpu.call_count,1)

    def test_gpu_rows_are_attributed_only_to_managed_pid(self):
        from allocator.runtime import parse_nvidia_telemetry
        global_rows = 'GPU-a, Example GPU, 1024, 8192\n'
        process_rows = '42, GPU-a, 256\n43, GPU-a, 700\n'
        result = parse_nvidia_telemetry(global_rows, process_rows, 42)
        self.assertEqual(result['gpu_process_bytes'], {'GPU-a':256*1024*1024})
        self.assertEqual(result['gpu_global'][0]['used_bytes'],1024*1024*1024)
        self.assertEqual(parse_nvidia_telemetry(global_rows,process_rows,None)['gpu_process_bytes'],{})

    def test_logs_are_bounded_redacted_and_metrics_extracted(self):
        from allocator.runtime import Runtime
        runtime = Runtime()
        for _ in range(1000):
            runtime._log('fixture '+('x'*9000))
        runtime._log('Authorization: Bearer NEVER_SHOW_ME')
        runtime._log('api_key=ALSO_SECRET token=SECRET_TOO')
        runtime._log('prompt eval time = 1 ms / 1 tokens ( 43.21 tokens per second)')
        runtime._log('eval time = 1 ms / 1 runs ( 12.34 tokens per second)')
        runtime._log('acceptance rate = 75%')
        state = runtime.snapshot()
        self.assertLessEqual(len(state['logs']),400)
        self.assertLessEqual(max(map(len,state['logs'])),2048)
        self.assertNotIn('NEVER_SHOW_ME',str(state))
        self.assertNotIn('ALSO_SECRET',str(state))
        self.assertEqual(state['prompt_tps'],43.21)
        self.assertEqual(state['generation_tps'],12.34)
        self.assertEqual(state['acceptance'],.75)

    def test_real_fixture_lifecycle_and_clean_environment(self):
        from allocator.runtime import Runtime
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp)/'fixture'
            executable.write_text('#!/usr/bin/python3\nimport os,time\nprint("fixture-start "+str(os.getenv("LLAMA_ARG_MODEL")),flush=True)\ntime.sleep(60)\n')
            executable.chmod(0o700)
            runtime = Runtime()
            os.environ['LLAMA_ARG_MODEL'] = 'MUST_NOT_INHERIT'
            try:
                runtime.start([str(executable)], {'host':'127.0.0.1','port':18997})
                wait_for(lambda: any('fixture-start None' in x for x in runtime.snapshot()['logs']))
                state = runtime.snapshot()
                self.assertTrue(state['running'])
                self.assertGreater(state['pid'], 0)
                self.assertGreater(state['rss_bytes'], 0)
                runtime.stop()
                self.assertFalse(runtime.snapshot()['running'])
                self.assertEqual(runtime.snapshot()['rss_bytes'], 0)
                self.assertEqual(runtime.snapshot()['gpu_process_bytes'], {})
            finally:
                runtime.stop()
                del os.environ['LLAMA_ARG_MODEL']


if __name__ == '__main__':
    unittest.main()
