"""Fixed degradation per held-out validation image; no training augmentation."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.data.data_util import training_paths
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RestorationValidationDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.paths = training_paths(opt['dataroot_gt'], opt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        gt = cv2.imread(path)
        if gt is None:
            raise ValueError(f'Unreadable validation image: {path}')
        gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_AREA)
        relative = Path(path).relative_to(Path(self.opt['dataroot_gt']).resolve()).as_posix()
        seed = int(hashlib.sha256(relative.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        size = int(rng.integers(85, 256))
        lq = cv2.GaussianBlur(gt, (5, 5), float(rng.uniform(.5, 2)))
        lq = cv2.resize(cv2.resize(lq, (size, size)), (512, 512), interpolation=cv2.INTER_CUBIC)
        lq = np.clip(lq.astype(np.float32) + rng.normal(0, 5, lq.shape), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode('.jpg', lq, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 76))])
        if not ok:
            raise RuntimeError('Validation JPEG encoding failed')
        lq = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        def tensor(img):
            return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float() / 127.5 - 1
        low = tensor(lq)
        return {'in': low, 'in_large_de': low, 'gt': tensor(gt), 'lq_path': path, 'gt_path': path}
