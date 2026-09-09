import importlib.util,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('inspection',R/'scripts/inspect_artifact.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ArtifactTests(unittest.TestCase):
 def test_trigger_shapes_support_and_distinctness(self):
  d=mod.summarize();self.assertEqual(len(d['ct_order']),5)
  self.assertEqual(next(x for x in d['triggers'] if x['id']=='CT2')['hamming_weight'],2)
 def test_manifest(self):self.assertGreater(mod.verify(),30)
 def test_no_oversized_files(self):
  self.assertFalse([p.name for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and p.stat().st_size>=100_000_000])
if __name__=='__main__':unittest.main()
