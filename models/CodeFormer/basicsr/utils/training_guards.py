"""Required pretrained networks must never be replaced by random teachers."""
from pathlib import Path


def teacher_options(options):
    config = dict(options['network_vqgan'])
    raw = config.get('model_path')
    if not raw:
        raise ValueError('network_vqgan.model_path is required for latent ground truth')
    path = Path(raw)
    if not path.is_absolute():
        root = Path(options.get('path', {}).get('root') or Path(__file__).resolve().parents[2])
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Pretrained VQGAN teacher not found: {path}')
    config['model_path'] = str(path)
    return config
