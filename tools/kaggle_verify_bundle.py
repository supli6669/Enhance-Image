"""Build a private, portable Kaggle GPU verification bundle; never upload it."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_dataset import digest, safe_path


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def extract_verified(archive, manifest, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new extraction directory')
    if hash_file(archive) != manifest['archive_sha256']:
        raise ValueError('Archive hash mismatch')
    expected = {row['name']: row for row in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate manifest entries')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(expected) or set(z.namelist()) != set(expected):
            raise ValueError('Archive members differ from manifest')
        for item in z.infolist():
            target = (destination/item.filename).resolve()
            if not target.is_relative_to(destination) or '\\' in item.filename or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe archive member')
            if item.file_size != expected[item.filename]['bytes']:
                raise ValueError('Archive file size mismatch')
        destination.mkdir(parents=True)
        for item in z.infolist():
            target = destination/item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256()
            with z.open(item) as source, target.open('xb') as out:
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    value.update(chunk)
                    out.write(chunk)
            if value.hexdigest() != expected[item.filename]['sha256']:
                raise ValueError('Extracted file hash mismatch; do not use partial extraction')


def unpack_or_copy_verified(dataset_root, manifest, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new extraction directory')
    zip_candidates = list(dataset_root.glob('kaggle_verify_payload.zip'))
    dir_candidates = list(dataset_root.glob('kaggle_verify_payload'))
    if zip_candidates:
        extract_verified(zip_candidates[0], manifest, destination)
    elif dir_candidates:
        source_dir = dir_candidates[0]
        destination.mkdir(parents=True)
        import shutil
        for row in manifest['files']:
            name = row['name']
            src = source_dir / name
            if not src.is_file() or src.stat().st_size != row['bytes']:
                raise ValueError(f'Missing or truncated file in dataset: {name}')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            if hash_file(target) != row['sha256']:
                raise ValueError(f'File hash mismatch: {name}')
    else:
        raise RuntimeError('Neither kaggle_verify_payload.zip nor kaggle_verify_payload directory found')


def notebook():
    bootstrap = '''import json, hashlib, zipfile, sys, subprocess, os, uuid
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Enable a Kaggle GPU accelerator before running this notebook')
manifest_candidates = list(Path('/kaggle/input').rglob('bundle_manifest.json'))
if len(manifest_candidates) != 1:
    raise RuntimeError('Attach exactly one private verification bundle dataset')
manifest_file = manifest_candidates[0]
dataset_root = manifest_file.parent
manifest = json.loads(manifest_file.read_text())
work = Path('/kaggle/working') / ('enhancer_verify_' + uuid.uuid4().hex[:10])
'''
    extract = inspect.getsource(hash_file)+'\n'+inspect.getsource(extract_verified)+'\n'+inspect.getsource(unpack_or_copy_verified)+'''\nunpack_or_copy_verified(dataset_root, manifest, work)
code = work/'code'
for path in (work/'weights').rglob('*.pth'):
    target = code/'weights'/path.relative_to(work/'weights')
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(path, target)
os.chdir(code)
print('Verified bundle:', len(manifest['files']), 'files')
'''
    install = '''# Keep the platform's working CUDA torch/torchvision pair; no CPU torch replacement.
requirements = ['numpy<2', 'addict==2.4.0', 'future==1.0.0', 'lmdb==1.5.1',
                'PyYAML==6.0.2', 'scikit-image==0.24.0', 'scipy==1.14.1',
                'yapf==0.40.2', 'facexlib==0.3.0', 'lpips==0.1.4',
                'opencv-python-headless==4.10.0.84', 'tqdm', 'requests', 'Pillow']
subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
args = ['--dataset-dir', str(work/'dataset'), '--split-dir', str(work/'split')]
subprocess.run([sys.executable, '-B', '-u', 'train_custom.py', '--preflight', *args], check=True)
'''
    run = '''# Exactly two optimizer iterations; no pilot, export, quantization or promotion.
command = [sys.executable, '-B', '-u', 'train_custom.py', '--verify', '--fresh', '--require-gpu', *args]
with (work/'gpu_verify.log').open('x') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError(f'GPU verification failed: inspect {work / "gpu_verify.log"}')
experiments = list((code/'models/CodeFormer/experiments').glob('*CodeFormer_gpu_verify_*'))
if len(experiments) != 1:
    raise RuntimeError('Expected exactly one isolated verification experiment')
experiment = experiments[0]
required = [experiment/'models/net_g_2.pth', experiment/'models/net_d_2.pth', experiment/'training_states/2.state']
if not all(p.is_file() and p.stat().st_size > 0 for p in required):
    raise RuntimeError('Iteration-2 checkpoint is incomplete')
# Only load state produced by this verified run.
state = torch.load(required[-1], map_location='cpu', weights_only=False)
if state['iter'] != 2 or not state['optimizers'] or not state['schedulers']:
    raise RuntimeError('Invalid optimizer/scheduler checkpoint state')
report = {'status':'gpu_two_iteration_check_passed', 'gpu':torch.cuda.get_device_name(0),
          'torch':torch.__version__, 'iterations':2, 'checkpoint_state_readable':True,
          'resume_execution_verified':False, 'production_approved':False,
          'bundle_sha256':manifest['archive_sha256'],
          'checkpoint_files':[{'name':p.name,'sha256':hash_file(p)} for p in required]}
(work/'gpu_verify_report.json').write_text(json.dumps(report, indent=2))
(Path('/kaggle/working')/'gpu_verify_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print('Evidence:', work)
'''
    cells = [{'cell_type':'markdown','metadata':{},'source':['# Private GPU verification — 2 iterations only\nAttach the generated private dataset and enable GPU + Internet. This notebook never starts full training.']}]
    for source in (bootstrap, extract, install, run):
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(keepends=True)})
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}


def build(root, split_dir, baseline, output):
    if output.exists():
        raise FileExistsError('Choose a new bundle directory')
    from tools.prepare_baseline import require_experimental_split
    from tools.training_preflight import configure_training
    import yaml
    split = json.loads((split_dir/'split.json').read_text(encoding='utf-8'))
    require_experimental_split(split, split_dir)
    report = json.loads(baseline.read_text(encoding='utf-8'))
    summary = report['summary']
    if summary['total_samples'] != summary['unique_reference_count'] or summary['total_samples'] < 300 or summary['evaluation']['scope'] != 'full':
        raise ValueError('A complete full baseline is required')
    if summary['evaluation'].get('split_sha256') != digest(split_dir/'split.json') or summary['manifest_sha256'] != split['benchmark_manifest_sha256']:
        raise ValueError('Baseline does not match the selected split')
    config = yaml.safe_load((ROOT/'models/CodeFormer/options/CodeFormer_stage3_custom.yml').read_text())
    _, preflight = configure_training(config, root, split_dir, split_dir/'holdout_paths.txt')
    members = {}
    def add(path, name):
        if name in members or not path.is_file():
            raise ValueError(f'Duplicate or missing bundle file: {name}')
        members[name] = path
    names = {r['path'] for group in ('train','validation') for r in split[group]}
    names.update(n for n in (split_dir/'holdout_paths.txt').read_text().splitlines() if n)
    for name in sorted(names):
        add(safe_path(root,name),'dataset/'+name)
    for name in ('split.json','train.txt','validation.txt','holdout_paths.txt','quality.json'):
        add(split_dir/name,'split/'+name)
    add(baseline,'baseline/report.json')
    for name in ('CodeFormer/codeformer.pth','facelib/vqgan_code1024.pth','facelib/recognition_arcface_ir_se50.pth'):
        add(ROOT/'weights'/name,'weights/'+name)
    # Whitelisted working source only; no git history, credentials, datasets or old checkpoints.
    tracked = subprocess.check_output(['git','ls-files','--','*.py','*.yml','*.yaml','requirements.txt'],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        if name.startswith(('tools/','models/CodeFormer/')) or '/' not in name:
            if any(part in ('datasets','experiments','weights') for part in Path(name).parts):
                continue
            add(ROOT/name,'code/'+name)
    for name in ('tools/kaggle_verify_config.py','tools/kaggle_verify_bundle.py'):
        if 'code/'+name not in members:
            add(ROOT/name,'code/'+name)
    output.mkdir(parents=True)
    archive = output/'kaggle_verify_payload.zip'
    records = []
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for index,(name,path) in enumerate(sorted(members.items())):
            records.append({'name':name,'bytes':path.stat().st_size,'sha256':hash_file(path)})
            z.write(path,name)
            if (index+1)%500==0:
                print(f'Packed {index+1}/{len(members)} files',flush=True)
    manifest={'schema_version':1,'purpose':'private_gpu_two_iteration_verification','files':records,
              'archive_sha256':hash_file(archive),'baseline_sha256':digest(baseline),
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_source_changes':bool(subprocess.check_output(['git','status','--porcelain','--','train_custom.py','tools'],cwd=ROOT,text=True).strip()),
              'preflight':preflight,'privacy':'private; contains portrait images; do not publish'}
    (output/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'gpu_verify.ipynb').write_text(json.dumps(notebook(),indent=2),encoding='utf-8')
    print(f'Bundle ready: {len(records)} files; {archive.stat().st_size} bytes',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'models/CodeFormer/datasets/ffhq/ffhq_512')
    p.add_argument('--split-dir',type=Path,default=ROOT/'benchmarks/splits/portraits_experiment_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'benchmarks/reports/experiment_baseline_run_v1/report.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    build(a.root,a.split_dir,a.baseline,a.output)
