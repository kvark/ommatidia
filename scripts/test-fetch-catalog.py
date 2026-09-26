#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('catalog', Path(__file__).with_name('fetch-catalog.py'))
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class GeometrySupport(unittest.TestCase):
    def check_primitives(self, primitives, modes=(), count=3):
        doc = {'materials': [{'alphaMode': mode} for mode in modes],
               'meshes': [{'primitives': primitives}], 'accessors': [{'count': count}]}
        raw = json.dumps(doc).encode()
        raw += b' ' * (-len(raw) % 4)
        glb = struct.pack('<5I', 0x46546C67, 2, 20 + len(raw), len(raw), 0x4E4F534A) + raw
        return catalog.has_opaque_triangles(glb)

    def test_capture_support(self):
        def check_modes(modes):
            return self.check_primitives(
                [{'material': i, 'indices': 0} for i in range(len(modes))], modes)
        self.assertTrue(check_modes(['OPAQUE']))
        self.assertTrue(check_modes(['BLEND', 'OPAQUE']))
        self.assertFalse(check_modes(['BLEND', 'MASK']))
        self.assertFalse(check_modes([]))

    def test_default_material_and_unindexed_geometry(self):
        self.assertTrue(self.check_primitives([{'indices': 0}]))
        self.assertTrue(self.check_primitives([{'attributes': {'POSITION': 0}}]))

    def test_empty_and_nontriangle_primitives(self):
        self.assertFalse(self.check_primitives([{}]))
        self.assertFalse(self.check_primitives([{'indices': 0}], count=2))
        self.assertFalse(self.check_primitives([{'indices': 0, 'mode': 1}]))


if __name__ == '__main__':
    unittest.main()
