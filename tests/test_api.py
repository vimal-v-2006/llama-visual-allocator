"""Control boundary tests; no inference is simulated."""
import unittest
from fastapi.testclient import TestClient


class SecurityTests(unittest.TestCase):
    def setUp(self):
        from allocator.app import create_app
        self.app = create_app(port=8095)
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8095')

    def test_host_origin_fetch_checks_precede_nonce_consumption(self):
        nonce = self.app.state.launch_nonce
        for headers in ({'Host':'evil.example'}, {'Origin':'http://127.0.0.1:8096'}, {'Sec-Fetch-Site':'cross-site'}, {'Host':'localhost:8095'}, {'Host':'127.0.0.1:8095','Origin':'null'}):
            self.assertEqual(self.client.post('/api/bootstrap', json={'nonce':nonce}, headers=headers).status_code, 403)
            self.assertEqual(self.app.state.launch_nonce, nonce)
        self.assertEqual(self.client.post('/api/bootstrap', json={'nonce':nonce}).status_code, 200)

    def test_json_and_body_bounds(self):
        self.assertEqual(self.client.post('/api/bootstrap', content='{}').status_code, 415)
        self.assertEqual(self.client.post('/api/bootstrap', content='x'*70000, headers={'Content-Type':'application/json'}).status_code, 413)
        self.assertEqual(self.client.post('/api/bootstrap', content='{', headers={'Content-Type':'application/json'}).status_code, 400)
        self.assertEqual(self.client.post('/api/bootstrap', json=[]).status_code, 400)

    def test_static_is_public_but_cannot_escape_root(self):
        from allocator.app import create_app
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'dist'
            root.mkdir()
            (root/'index.html').write_text('<h1>fixture UI</h1>')
            outside = Path(temp)/'private.txt'
            outside.write_text('PRIVATE')
            (root/'symlink').symlink_to(outside)
            client = TestClient(create_app(port=8095, static_dir=root), base_url='http://127.0.0.1:8095')
            self.assertEqual(client.get('/').text, '<h1>fixture UI</h1>')
            for path in ('/symlink','/%2e%2e/private.txt','/api/unknown'):
                response = client.get(path)
                self.assertNotEqual(response.status_code, 200)
                self.assertNotIn('PRIVATE', response.text)
            self.assertIn('default-src', client.get('/').headers['content-security-policy'])

    def test_nonce_is_one_use_and_not_api_authority(self):
        nonce = self.app.state.launch_nonce
        self.assertEqual(self.client.get('/api/runtime', headers={'Authorization': 'Bearer '+nonce}).status_code, 403)
        response = self.client.post('/api/bootstrap', json={'nonce': nonce})
        self.assertEqual(response.status_code, 200)
        token = response.json()['token']
        self.assertNotEqual(token, nonce)
        self.assertNotIn('set-cookie', response.headers)
        self.assertEqual(self.client.post('/api/bootstrap', json={'nonce': nonce}).status_code, 403)
        self.assertEqual(self.client.get('/api/runtime', headers={'Authorization':'Bearer '+token}).status_code, 200)


class ContractTests(unittest.TestCase):
    def test_engine_contract_and_fresh_start_checks(self):
        # Explicit control-contract fixture, never an inference implementation.
        from allocator.app import create_app
        import copy
        import sys
        from pathlib import Path
        executable = str(Path('/usr/bin/python3').resolve())
        class EngineFixture:
            calls = []
            def default_config(self, caps=None):
                return {'executable':executable, 'model_path':'/model.gguf', 'server':{'host':'127.0.0.1','port':8096}}
            def probe(self, path):
                self.calls.append('probe')
                return {'executable':path}
            def read_model(self, path):
                self.calls.append('model')
                return {'path':path}
            def plan(self, model, caps, config, mode=None):
                self.calls.append('plan')
                return {'config':copy.deepcopy(config), 'status':'OOM', 'launchable':False, 'argv':[sys.executable], 'errors':['fixture OOM']}
        engine = EngineFixture()
        app = create_app(port=8095, engine_module=engine)
        with TestClient(app, base_url='http://127.0.0.1:8095') as client:
            token = client.post('/api/bootstrap', json={'nonce':app.state.launch_nonce}).json()['token']
            client.headers['Authorization'] = 'Bearer '+token
            self.assertEqual(client.get('/api/state').status_code, 200)
            self.assertEqual(client.post('/api/probe', json={'executable':executable}).status_code, 200)
            self.assertEqual(client.post('/api/model', json={'path':'/model.gguf'}).status_code, 200)
            self.assertEqual(client.post('/api/plan', json={'config':engine.default_config()}).json()['status'], 'OOM')
            engine.calls.clear()
            self.assertEqual(client.post('/api/start', json={'config':engine.default_config()}).status_code, 409)
            self.assertEqual(engine.calls, ['probe','model','plan'])
            bad = engine.default_config()
            bad['server']['host'] = '0.0.0.0'
            self.assertEqual(client.post('/api/start', json={'config':bad}).status_code, 400)
            self.assertFalse(client.post('/api/stop', json={}).json()['running'])


class LaunchPolicyTests(unittest.TestCase):
    def test_unknown_ack_network_and_typed_sections(self):
        from allocator.app import create_app
        from unittest.mock import patch
        from pathlib import Path
        class Engine:
            status = 'UNKNOWN'; errors = []
            def default_config(self, caps=None):
                return dict(executable=str(Path('/usr/bin/python3').resolve()),model_path='/fixture', server={'host':'0.0.0.0','port':8096,'allow_network':True},acknowledge_estimate=True,placement_mode='manual',split={},sampling={},reasoning={},vision={},environment={},context={'kv_offload':False})
            def probe(self, path): return {}
            def read_model(self, path): return {}
            def plan(self, model, caps, config, mode=None):
                return dict(config=config,status=self.status,errors=self.errors,launchable=True,argv=[config['executable']],env={'GGML_CUDA_DISABLE_GRAPHS':'1'})
        engine=Engine(); app=create_app(engine_module=engine)
        with TestClient(app,base_url='http://127.0.0.1:8095') as client, patch.object(app.state.runtime,'start',return_value={'running':True}) as start:
            token=client.post('/api/bootstrap',json={'nonce':app.state.launch_nonce}).json()['token']
            client.headers['Authorization']='Bearer '+token
            cfg=engine.default_config()
            response=client.post('/api/start',json={'config':cfg})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(start.call_args.kwargs['env'],{'GGML_CUDA_DISABLE_GRAPHS':'1'})
            planned=client.post('/api/plan',json={'config':cfg}).json()
            self.assertTrue(any('network' in w.lower() for w in planned['warnings']))
            for ack in (False,1,'true',None):
                cfg['acknowledge_estimate']=ack
                self.assertNotEqual(client.post('/api/start',json={'config':cfg}).status_code,200)
            cfg['acknowledge_estimate']=True
            for status,errors in [('OOM',[]),('UNKNOWN',['invalid'])]:
                engine.status=status; engine.errors=errors
                self.assertEqual(client.post('/api/start',json={'config':cfg}).status_code,409)
            for allow in (False,1,'true',None):
                cfg['server']['allow_network']=allow
                self.assertEqual(client.post('/api/start',json={'config':cfg}).status_code,400)


class CacheTests(unittest.TestCase):
    def test_live_plans_cache_stat_identity_but_start_is_fresh(self):
        import tempfile, copy
        from pathlib import Path
        from allocator.app import create_app
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp)/'server'; exe.write_text('#!/bin/sh\nexit 0\n'); exe.chmod(0o700)
            model = Path(tmp)/'model.gguf'; model.write_bytes(b'fixture')
            shard = Path(tmp)/'other.gguf'; shard.write_bytes(b'shard')
            class Engine:
                probes = reads = 0
                def default_config(self, caps=None):
                    return dict(executable=str(exe), model_path=str(model), devices={'RAM':{'enabled':True}} if caps else {}, server={'host':'127.0.0.1','port':8096})
                def probe(self, path):
                    self.probes += 1; return {'executable':path}
                def read_model(self, path):
                    self.reads += 1; return {'path':path,'shards':[path,str(shard)]}
                def plan(self, model, caps, config, mode=None):
                    return dict(config=copy.deepcopy(config), status='OOM', errors=[], launchable=False, argv=[])
            engine = Engine(); app = create_app(engine_module=engine)
            with TestClient(app, base_url='http://127.0.0.1:8095') as client:
                token = client.post('/api/bootstrap',json={'nonce':app.state.launch_nonce}).json()['token']
                client.headers['Authorization'] = 'Bearer '+token
                loaded = client.post('/api/model',json={'path':str(model)}).json()
                cfg = loaded['config']
                self.assertTrue(cfg['devices']['RAM']['enabled'])
                for _ in range(3):
                    self.assertEqual(client.post('/api/plan',json={'config':cfg}).status_code,200)
                self.assertEqual((engine.probes,engine.reads),(1,1))
                shard.write_bytes(b'changed shard')
                client.post('/api/plan',json={'config':cfg})
                self.assertEqual(engine.reads,2)
                exe.write_text('#!/bin/sh\nexit 1\n')
                client.post('/api/plan',json={'config':cfg})
                self.assertEqual(engine.probes,2)
                client.post('/api/start',json={'config':cfg})
                self.assertEqual((engine.probes,engine.reads),(3,3))
                cfg.update(placements={'old':'RAM'},locked=['old'],positions={'old':{'x':1}},edges=[{'source':'old'}],compute_mib=123)
                client.post('/api/plan',json={'config':cfg})
                other = Path(tmp)/'new.gguf'; other.write_bytes(b'new')
                changed = client.post('/api/model',json={'path':str(other)}).json()['config']
                self.assertEqual(changed['placements'],{})
                self.assertEqual(changed['locked'],[])
                self.assertEqual(changed['compute_mib'],123)


class CliTests(unittest.TestCase):
    def test_cli_serves_authenticated_loopback(self):
        import subprocess, sys, time, socket
        import httpx
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port = sock.getsockname()[1]
        process = subprocess.Popen([sys.executable,'-m','allocator.app','--port',str(port),'--no-browser'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = process.stdout.readline().strip()
            self.assertTrue(line.startswith(f'http://127.0.0.1:{port}/#launch='), 'Missing launch URL')
            nonce = line.split('#launch=')[1]
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False) as client:
                for _ in range(100):
                    try:
                        response = client.post('/api/bootstrap',json={'nonce':nonce})
                        break
                    except httpx.ConnectError:
                        time.sleep(.05)
                else:
                    self.fail('CLI did not serve')
                self.assertEqual(response.status_code,200)
                self.assertEqual(client.get('/api/runtime').status_code,403)
                self.assertEqual(client.get('/api/runtime',headers={'Authorization':'Bearer '+response.json()['token']}).status_code,200)
        finally:
            process.terminate()
            process.wait(timeout=15)
            process.stdout.close()
            process.stderr.close()


if __name__ == '__main__':
    unittest.main()
