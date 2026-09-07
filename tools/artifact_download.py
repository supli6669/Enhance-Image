"""Stage remote artifacts without overwriting checkpoints or trusting remote paths."""
from pathlib import Path, PurePosixPath
import requests


def destination(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts or ':' in name:
        raise ValueError('Unsafe artifact path')
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError('Artifact path escapes staging directory')
    return target


def download(url, target):
    target = Path(target)
    partial = target.with_name(target.name + '.download')
    if target.exists() or partial.exists():
        raise FileExistsError(f'Choose a new staging directory: {target.name} already exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(20, 120), headers={'Accept-Encoding': 'identity'}) as response:
        if response.status_code != 200:
            raise RuntimeError(f'Artifact download HTTP {response.status_code}')
        expected = int(response.headers.get('content-length', 0))
        count = 0
        with partial.open('xb') as handle:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    count += len(chunk)
        if count == 0 or (expected and count != expected):
            raise ValueError(f'Incomplete artifact: {target.name}; kept .download file')
    partial.rename(target)
    return target
