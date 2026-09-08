"""Bounded GPU smoke configuration; never changes production YAML or old runs."""
from pathlib import Path
import uuid


def configure_verify(config, project_root):
    run_id = 'CodeFormer_gpu_verify_' + uuid.uuid4().hex[:12]
    directory = Path(project_root) / 'artifacts' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    validation = config['datasets']['val']
    paths = Path(validation['image_manifest']).read_text(encoding='utf-8').splitlines()
    if len(paths) < 2:
        raise ValueError('Verification requires two real validation images')
    manifest = directory / 'validation.txt'
    manifest.write_text('\n'.join(paths[:2])+'\n', encoding='utf-8')
    validation['image_manifest'] = str(manifest)
    config['name'] = run_id
    config['verification_only'] = True
    config['train']['total_iter'] = 2
    config['logger'].update(print_freq=1, save_checkpoint_freq=2, use_tb_logger=False)
    config['val']['val_freq'] = 2
    config['datasets']['train'].update(batch_size_per_gpu=1, num_worker_per_gpu=0)
    return directory


def check_finite_step(model):
    import math
    import torch
    losses = model.get_current_log()
    if not losses or not all(math.isfinite(float(value)) for value in losses.values()):
        raise RuntimeError('Verification found missing or non-finite losses')
    gradients = [p.grad for name in ('net_g', 'net_d') for p in getattr(model, name).parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError('Verification found missing or non-finite gradients')
