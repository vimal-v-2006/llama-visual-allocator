import copy
import unittest
from allocator import engine
from test_engine_plan import SMALL
from test_engine_probe import EXE


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caps = engine.probe(EXE)
        for d in cls.caps['devices']:
            d['free_bytes'] = d['total_bytes']  # hermetic: engine-semantics tests must not depend on live GPU occupancy
        cls.model = engine.read_model(SMALL)

    def test_malformed_control_shapes_fail_closed(self):
        cases=[{'vision':{'model_path':5,'enabled':True}}, {'spec':{'device':[]}},
               {'devices':{'CUDA0':None},'split':{'devices':['CUDA0']}},
               {'server':{'host':123,'allow_network':True}},
               {'sampling':{'unknown':3}}, {'reasoning':{'unknown':True}}]
        for case in cases:
            with self.subTest(case=case):
                self.assertTrue(engine.plan(self.model,self.caps,case)['errors'])

    def test_projector_parsed_weights_reserved_and_flag_gated(self):
        path='/home/vimal/Documents/Models/unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf'
        c=engine.default_config(self.caps)
        c['vision'].update(enabled=True,model_path=path,device='RAM')
        p=engine.plan(self.model,self.caps,c)
        self.assertFalse(p['errors'],p['errors'])
        self.assertIn('--mmproj',p['argv'])
        self.assertIn('--no-mmproj-offload',p['argv'])
        self.assertEqual(next(r for r in p['memory'] if r['id']=='RAM')['vision'],931126208)
        self.assertEqual(p['status'],'UNKNOWN')
        caps=copy.deepcopy(self.caps)
        caps['flags'].remove('--mmproj')
        self.assertTrue(engine.plan(self.model,caps,c)['errors'])
        c['vision']['model_path']=SMALL
        self.assertTrue(engine.plan(self.model,self.caps,c)['errors'])

    def test_hybrid_real_model_uses_only_attention_layers_for_kv(self):
        path='/home/vimal/Documents/Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'
        model=engine.read_model(path)
        self.assertEqual(model['weight_bytes'],17548181504)
        c=engine.default_config(self.caps)
        c['context'].update(size=131072,k_type='q4_0',v_type='q4_0',flash='on')
        kv, known=engine._kv_bytes(model,c['context'])
        self.assertFalse(known)
        full_layers={t['layer'] for t in model['tensors'] if '.attn_k.weight' in t['name']}
        expected=len(full_layers)*2*(256*4//32*18)*131072
        self.assertEqual(kv,expected, 'recurrent state is unknown, not an invented full-attention cache')
        c['spec'].update(mode='draft-mtp',model_path='')
        c['placement_mode']='layer'
        c['split'].update(devices=['CUDA0','Vulkan1'],ratios=[11,6])
        c['context']['kv_offload']=True
        p=engine.plan(model,self.caps,c)
        self.assertFalse(p['errors'],p['errors'])
        self.assertEqual(p['status'],'UNKNOWN')
        self.assertNotIn('--spec-draft-model',p['argv'])
        self.assertFalse(p['launchable'])
        self.assertTrue(any('unknown' in w for w in p['warnings']))
        c['acknowledge_estimate']=True
        acknowledged=engine.plan(model,self.caps,c)
        self.assertTrue(acknowledged['launchable'],acknowledged['errors'])
        self.assertEqual(acknowledged['status'],'UNKNOWN')
        c['server']['port']=-1
        self.assertFalse(engine.plan(model,self.caps,c)['launchable'])

    def test_layer_split_preserves_ratios_and_sums_real_groups(self):
        c = engine.default_config(self.caps)
        c.update(placement_mode='layer', split=dict(mode='layer',ratios=[11,6],main_gpu=0,gpu_layers=10,devices=['CUDA0','Vulkan1']))
        c['context'].update(kv_offload=True, k_type='q4_0', v_type='q4_0', flash='on')
        c['placements'] = {self.model['groups'][0]['id']:'RAM'}
        p = engine.plan(self.model,self.caps,c)
        self.assertFalse(p['errors'],p['errors'])
        self.assertIn('--split-mode',p['argv'])
        self.assertEqual(p['argv'][p['argv'].index('--tensor-split')+1],'11,6')
        self.assertEqual(p['argv'][p['argv'].index('--gpu-layers')+1],'10')
        self.assertNotIn('--override-tensor',p['argv'])
        self.assertNotIn('--no-kv-offload',p['argv'])
        self.assertIn('--kv-offload',p['argv'])
        self.assertEqual(p['config']['placements'],c['placements'])
        offloaded=set()
        for g in self.model['groups']:
            if p['allocation'][g['id']] != 'RAM' and g['layer'] is not None:
                offloaded.add(g['layer'])
        self.assertEqual(len(offloaded),9)  # installed build counts output as a GPU layer
        for row in p['memory']:
            self.assertEqual(row['weights'],sum(g['nbytes'] for g in self.model['groups'] if p['allocation'][g['id']]==row['id']))
        self.assertGreater(sum(r['kv'] for r in p['memory'] if r['id']!='RAM'),0)
        for layer in offloaded:
            self.assertEqual(len({p['allocation'][g['id']] for g in self.model['groups'] if g['layer']==layer}),1)

    def test_typed_controls_reject_invalid_values(self):
        for changes in [dict(sampling={'top_p':2}), dict(sampling={'temperature':True}),
                        dict(reasoning={'jinja':'yes'}), dict(reasoning={'effort':'invented'}),
                        dict(environment={'cuda_disable_graphs':'1'}),
                        dict(server={'host':'bad.sock', 'allow_network':True}),
                        dict(acknowledge_estimate='yes'), dict(context={'kv_offload':'yes'}),
                        dict(vision={'enabled':'yes'}), dict(placement_mode='row'),
                        dict(split={'ratios':[0,0]}), dict(split={'devices':['FAKE']}),
                        dict(split={'gpu_layers':-1}), dict(split={'main_gpu':True})]:
            with self.subTest(changes=changes):
                self.assertTrue(engine.plan(self.model,self.caps,changes)['errors'])

    def test_optional_controls_compile_and_gate(self):
        c = engine.default_config(self.caps)
        c.update(sampling=dict(temperature=1, top_p=.95, top_k=20, min_p=0,
                               presence_penalty=0, repeat_penalty=1),
                 reasoning=dict(jinja=True, effort='xhigh', preserve=True),
                 environment=dict(cuda_disable_graphs=True))
        c['server'].update(host='0.0.0.0', allow_network=True)
        p = engine.plan(self.model, self.caps, c)
        self.assertFalse(p['errors'], p['errors'])
        for flag in ('--temp','--top-p','--top-k','--min-p','--presence-penalty',
                     '--repeat-penalty','--jinja','--reasoning-effort','--reasoning-preserve'):
            self.assertIn(flag, p['argv'])
            caps = copy.deepcopy(self.caps)
            caps['flags'].remove(flag)
            self.assertTrue(engine.plan(self.model, caps, c)['errors'])
        self.assertEqual(p['env'], {'GGML_CUDA_DISABLE_GRAPHS':'1'})
        base = engine.plan(self.model, self.caps, {})
        self.assertEqual(base['env'], {})
        self.assertNotIn('--temp', base['argv'])
        self.assertNotIn('--reasoning-effort', base['argv'])
