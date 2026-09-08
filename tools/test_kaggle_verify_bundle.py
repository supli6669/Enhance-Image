"""CPU-only tests for portable GPU verification artifacts and failure guards."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.kaggle_verify_bundle import extract_verified, hash_file, notebook
from tools.kaggle_verify_config import configure_verify, check_finite_step


class BundleTests(unittest.TestCase):
    def archive(self,root,name='code/a.py'):
        path=root/'payload.zip'
        data=b'print("fixture")\n'
        with zipfile.ZipFile(path,'x') as z:
            z.writestr(name,data)
        return path,{'archive_sha256':hash_file(path),'files':[{'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}

    def test_archive_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive,manifest=self.archive(root)
            extract_verified(archive,manifest,root/'output')
            self.assertTrue((root/'output/code/a.py').is_file())
            with self.assertRaises(FileExistsError):
                extract_verified(archive,manifest,root/'output')

    def test_zip_path_traversal_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive,manifest=self.archive(root,'../escape.py')
            with self.assertRaises(ValueError):
                extract_verified(archive,manifest,root/'output')
            self.assertFalse((root/'escape.py').exists())
            self.assertFalse((root/'output').exists())

    def test_corrupted_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive,manifest=self.archive(root)
            manifest['archive_sha256']='wrong'
            with self.assertRaises(ValueError):
                extract_verified(archive,manifest,root/'output')

    def test_verify_config_is_bounded_and_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'val.txt'; source.write_text('a.png\nb.png\nc.png\n')
            config={'name':'production','datasets':{'train':{},'val':{'image_manifest':str(source)}},'train':{'total_iter':6000},'logger':{},'val':{}}
            first=configure_verify(config,root)
            second=configure_verify(copy.deepcopy(config),root)
            self.assertNotEqual(first,second)
            self.assertEqual(source.read_text(),'a.png\nb.png\nc.png\n')
            self.assertEqual((first/'validation.txt').read_text(),'a.png\nb.png\n')
            self.assertEqual(config['train']['total_iter'],2)
            self.assertEqual(config['logger']['save_checkpoint_freq'],2)
            self.assertEqual(config['datasets']['train']['batch_size_per_gpu'],1)

    def test_nonfinite_gradient_or_loss_fails(self):
        g=torch.nn.Linear(1,1); d=torch.nn.Linear(1,1)
        g(torch.ones(1,1)).sum().backward()
        model=SimpleNamespace(net_g=g,net_d=d,get_current_log=lambda:{'loss':1.})
        check_finite_step(model)
        g.weight.grad.fill_(float('nan'))
        with self.assertRaises(RuntimeError):
            check_finite_step(model)

    def test_notebook_is_clean_and_gpu_verification_only(self):
        nb=notebook()
        combined=''
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source']); ast.parse(source)
                combined+=source
                self.assertEqual(cell['outputs'],[])
        self.assertIn("'--require-gpu'",combined)
        self.assertIn("'--verify'",combined)
        self.assertNotIn('rmtree',combined)
        self.assertNotIn('git clone',combined)


if __name__=='__main__':
    unittest.main()
