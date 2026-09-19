import importlib
import unittest
from pathlib import Path

MODEL = '/home/vimal/models/Qwen3.8-Flash-Next-GSQ-RCO/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'

class ModelTests(unittest.TestCase):
    @unittest.skipUnless(Path(MODEL).exists(), 'host integration model unavailable')
    def test_real_shards_are_counted_exactly_once(self):
        self.assertIsNotNone(importlib.util.find_spec('allocator.engine'), 'GGUF engine missing')
        from allocator.engine import read_model
        model = read_model(MODEL)
        self.assertEqual(len(model['shards']), 2)
        self.assertGreater(model['weight_bytes'], 20_000_000_000)
        self.assertEqual(sum(t['nbytes'] for t in model['tensors']), model['weight_bytes'])
        self.assertEqual(sum(g['nbytes'] for g in model['groups']), model['weight_bytes'])
        names = [t['name'] for t in model['tensors']]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(any(t['type'] == 'IQ3_XXS' for t in model['tensors']))
        self.assertEqual(model['file_bytes'], sum(Path(p).stat().st_size for p in model['shards']))
