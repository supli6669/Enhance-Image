---
name: lmdb-dataset-binary-storage
description: Converting loose image folders to Lightning Memory-Mapped Database (LMDB) binary datasets for ultra-fast sequential training I/O and zero disk fragmentation.
---

# LMDB Dataset Binary Storage & High-Speed I/O Skill

## Overview
Reading thousands of small loose image files during neural network training causes severe disk I/O bottlenecks and OS page cache thrashing. LMDB maps binary image buffers directly into virtual memory for instant zero-copy reads.

## Build LMDB Script Pattern
```python
import lmdb
import cv2

def build_lmdb(image_paths, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1099511627776) # 1TB max allocation
    with env.begin(write=True) as txn:
        for idx, path in enumerate(image_paths):
            with open(path, 'rb') as f:
                img_bin = f.read()
            key = f"{idx:08d}".encode('ascii')
            txn.put(key, img_bin)
    env.close()
```
