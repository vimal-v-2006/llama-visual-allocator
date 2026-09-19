import unittest
from pathlib import Path
from allocator import engine

EXE = '/home/vimal/AI/llama.cpp/build/bin/llama-server'

class ProbeTests(unittest.TestCase):
    @unittest.skipUnless(Path(EXE).exists(), 'installed llama.cpp unavailable')
    def test_live_flags_buffers_and_aliases(self):
        self.assertTrue(hasattr(engine, 'probe'), 'capability probe missing')
        c = engine.probe(EXE)
        self.assertIn('--override-tensor', c['flags'])
        self.assertNotIn('--draft-max', c['flags'])
        self.assertNotIn('--defrag-thold', c['flags'])
        self.assertIn('f16', c['cache_types'])
        devices = {d['id']:d for d in c['devices']}
        self.assertEqual(devices['CUDA0']['buffer_type'], 'CUDA0')
        self.assertEqual(devices['Vulkan0']['alias_of'], 'CUDA0')
        self.assertFalse(devices['Vulkan0']['available'])
        self.assertEqual(devices['MMAP']['alias_of'], 'RAM')
        self.assertIn('CPU', c['buffer_types'])
        self.assertTrue(c['version'])
