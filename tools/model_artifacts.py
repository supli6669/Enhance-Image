"""Inspect model provenance without loading tensors or changing active weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _external_references(path: str, modified: int, size: int) -> tuple:
    import onnx
    from onnx.external_data_helper import _get_all_tensors
    graph = onnx.load(path, load_external_data=False)
    return tuple(tuple((item.key, item.value) for item in tensor.external_data)
                 for tensor in _get_all_tensors(graph)
                 if tensor.data_location == onnx.TensorProto.EXTERNAL)


def model_files(path: Path) -> list[Path]:
    """Reject missing, truncated and cross-model external weight references."""
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or incomplete model: {path.name}')
    files = [path]
    if path.suffix == '.onnx':
        stat = path.stat()
        try:
            references = _external_references(str(path), stat.st_mtime_ns, stat.st_size)
        except Exception as error:
            raise ValueError(f'Unreadable ONNX graph: {path.name}') from error
        for reference in references:
            info = dict(reference)
            location = info.get('location', '')
            if location != path.name + '.data':
                raise ValueError(f'{path.name} references a different model sidecar: {location}')
            sidecar = path.parent / location
            offset, length = int(info.get('offset', 0)), int(info.get('length', 0))
            if offset < 0 or length <= 0 or not sidecar.is_file() or sidecar.stat().st_size < offset + length:
                raise ValueError(f'Missing or truncated sidecar for {path.name}')
            if sidecar not in files:
                files.append(sidecar)
    return files


@lru_cache(maxsize=4)
def _verified_baseline(model, model_stat, source, source_stat, manifest, manifest_stat):
    try:
        data = json.loads(Path(manifest).read_text(encoding='utf-8'))
        if data.get('inference_parity', {}).get('passed') is not True:
            return False
        if data['source']['files'][0]['sha256'] != sha256_file(Path(source)):
            return False
        records = {record['name']: record['sha256'] for record in data['model']['files']}
        return all(records.get(p.name) == sha256_file(p) for p in model_files(Path(model)))
    except (ValueError, KeyError, OSError):
        return False


def preferred_baseline(weights_dir):
    """Only a hash-matched parity-tested export may replace the PyTorch baseline."""
    root = Path(weights_dir)
    source = root / 'codeformer.pth'
    model = root / 'codeformer_baseline.onnx'
    manifest = model.with_suffix('.manifest.json')
    if all(p.is_file() for p in (source, model, manifest)):
        def signature(path):
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)
        if _verified_baseline(str(model), signature(model), str(source), signature(source),
                              str(manifest), signature(manifest)):
            return str(model)
    return str(source)


def describe_model(path: Path) -> dict:
    files = model_files(path)
    return {'path': str(files[0]), 'files': [
        {'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files
    ], 'quality_status': 'unvalidated'}


def inventory(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and (path.suffix in {'.onnx', '.pth'} or path.name.endswith('.download')):
            try:
                if path.name.endswith('.download'):
                    raise ValueError('Incomplete download; not eligible for inference')
                result.append(describe_model(path))
            except (ValueError, OSError) as error:
                result.append({'path': str(path), 'error': str(error), 'quality_status': 'invalid'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('weights/CodeFormer'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory(args.root), indent=2), encoding='utf-8')
    print(f'Model inventory saved to {args.output}')
