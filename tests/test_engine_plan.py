import unittest
from pathlib import Path
from allocator import engine
from test_engine_model import MODEL
from test_engine_probe import EXE

SMALL = str(Path(__file__).parent / 'models/SmolLM2-135M-Instruct-Q4_K_M.gguf')

class PlanTests(unittest.TestCase):
    @unittest.skipUnless(Path(SMALL).exists(), 'real small model unavailable')
    def test_draft_reserves_real_weights_before_allocation(self):
        caps, model = engine.probe(EXE), engine.read_model(SMALL)
        c = engine.default_config(caps)
        c['spec'].update(mode='draft-simple', model_path=SMALL, device='CUDA0')
        c['devices']['CUDA0']['cap_mib'] = 1300
        p = engine.plan(model,caps,c,'maximum')
        self.assertIn('--spec-draft-model',p['argv'])
        self.assertIn('--spec-draft-n-max',p['argv'])
        self.assertNotIn('--draft-max',p['argv'])
        row = next(x for x in p['memory'] if x['id']=='CUDA0')
        self.assertGreater(row['draft'],model['weight_bytes'])
        self.assertEqual(row['weights'],0, 'draft reservation should leave no room for main weights')
        self.assertTrue(any('compatibility' in w for w in p['warnings']))
        c['spec']['mode'] = 'draft-mtp'
        self.assertFalse(engine.plan(model,caps,c)['launchable'], 'SmolLM2 has no MTP tensors')

    @unittest.skipUnless(Path(SMALL).exists(), 'real small model unavailable')
    def test_mmap_cap_constrains_same_ram_pool(self):
        caps, model = engine.probe(EXE), engine.read_model(SMALL)
        c = engine.default_config(caps)
        c['placements'] = {g['id']:'MMAP' for g in model['groups']}
        c['devices']['MMAP']['cap_mib'] = 1
        p = engine.plan(model,caps,c)
        self.assertEqual(p['status'],'OOM')
        self.assertFalse(p['launchable'])
        self.assertEqual(sum(x['weights'] for x in p['memory']),model['weight_bytes'])
        self.assertEqual(next(x for x in p['memory'] if x['id']=='RAM')['mapped_bytes'],model['weight_bytes'])

    @unittest.skipUnless(Path(MODEL).exists(), 'real Qwen model unavailable')
    def test_qwen_real_weights_oom_and_hybrid_uncertainty(self):
        caps, model = engine.probe(EXE), engine.read_model(MODEL)
        self.assertEqual(model['weight_bytes'],75828974080)
        self.assertEqual(len(model['tensors']),1224)
        c = engine.default_config(caps)
        p = engine.plan(model,caps,c,'maximum')
        self.assertEqual(p['status'],'OOM')
        self.assertFalse(p['launchable'])
        self.assertEqual(sum(x['weights'] for x in p['memory']),model['weight_bytes'])
        self.assertTrue(any('not fully modeled' in w for w in p['warnings']))
        self.assertGreater(next(x for x in p['memory'] if x['id']=='RAM')['kv'],0)

    @unittest.skipUnless(Path(SMALL).exists(), 'real small model unavailable')
    def test_invalid_config_fails_closed(self):
        import copy
        caps, model = engine.probe(EXE), engine.read_model(SMALL)
        base = engine.default_config(caps)
        gid = model['groups'][0]['id']
        cases = [ {'placements':{gid:'FAKE0'}}, {'placements':{'invented-expert-slice':'RAM'}},
            {'locked':['invented']}, {'context':{'size':-1}}, {'context':{'k_type':'madeup'}},
            {'context':{'parallel':0}}, {'context':{'ubatch':9000}},
            {'overhead_mib':float('nan')}, {'compute_mib':-1},
            {'server':{'host':'0.0.0.0'}}, {'server':{'port':70000}},
            {'spec':{'mode':'invented'}}, {'executable':'/bin/true'},
            {'placements':{gid:'Vulkan0'}}, {'devices':{'FAKE0':{'enabled':True,'cap_mib':100}}} ]
        for changes in cases:
            c = copy.deepcopy(base)
            for k,v in changes.items():
                if isinstance(v,dict): c[k].update(v)
                else: c[k]=v
            with self.subTest(changes=changes):
                p = engine.plan(model,caps,c)
                self.assertFalse(p['launchable'])
                self.assertTrue(p['errors'])
        c = copy.deepcopy(base)
        c['placements'][gid] = 'CUDA0'
        c['devices']['CUDA0']['enabled'] = False
        self.assertFalse(engine.plan(model,caps,c)['launchable'])

    @unittest.skipUnless(Path(SMALL).exists(), 'real small model unavailable')
    def test_allocator_reserves_first_preserves_lock_and_compiles_exact_routes(self):
        caps = engine.probe(EXE)
        model = engine.read_model(SMALL)
        config = engine.default_config(caps)
        config['model_path'] = SMALL
        locked = model['groups'][0]['id']
        config['placements'][locked] = 'MMAP'
        config['locked'] = [locked]
        config['devices']['CUDA0']['cap_mib'] = 800
        # 768MiB reserved leaves only 32MiB for weights.
        p = engine.plan(model, caps, config, 'maximum')
        self.assertEqual(p['allocation'][locked], 'MMAP')
        self.assertIn('CUDA0', set(p['allocation'].values()))
        cuda = next(x for x in p['memory'] if x['id']=='CUDA0')
        self.assertLessEqual(cuda['weights'], 32 * engine.MIB)
        self.assertIn('--override-tensor', p['argv'])
        self.assertNotIn('Vulkan0', set(p['allocation'].values()))
        self.assertEqual(sum(x['weights'] for x in p['memory']), model['weight_bytes'])
        overrides = p['argv'][p['argv'].index('--override-tensor')+1]
        import re
        for t in model['tensors']:
            matches = [route for route in overrides.split(',') if re.search(route.rsplit('=',1)[0], t['name'])]
            self.assertEqual(len(matches), 1, t['name'])
        self.assertEqual(config['placements'], {locked:'MMAP'}, 'plan mutated input')

    @unittest.skipUnless(Path(SMALL).exists(), 'real small model unavailable')
    def test_real_model_cpu_plan_compiles_supported_command(self):
        self.assertTrue(hasattr(engine, 'default_config'), 'default config missing')
        caps = engine.probe(EXE)
        config = engine.default_config(caps)
        model = engine.read_model(SMALL)
        config['model_path'] = SMALL
        p = engine.plan(model, caps, config)
        self.assertTrue(p['launchable'], p['errors'])
        self.assertEqual(len(p['allocation']), len(model['groups']))
        self.assertEqual(sum(x['weights'] for x in p['memory']), model['weight_bytes'])
        self.assertIn('--load-mode', p['argv'])
        self.assertIn('--no-kv-offload', p['argv'])
        self.assertNotIn('--mmap', p['argv'])
        self.assertGreater(next(x for x in p['memory'] if x['id']=='RAM')['kv'], 0)
        self.assertEqual(p['argv'][p['argv'].index('--fit')+1], 'off')
        self.assertTrue(set(a for a in p['argv'] if a.startswith('--')).issubset(caps['flags']))
