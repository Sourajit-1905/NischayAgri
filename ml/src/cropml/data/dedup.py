"""
Perceptual-hash near-duplicate clustering.

Runs per class, before any augmentation. Clusters become the grouping unit
for leakage-free splitting in splitting.py: an entire cluster goes to one
split, never split across train/val/test.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass

import imagehash
from PIL import Image, UnidentifiedImageError

from cropml.data.loaders.base import Sample


class ImageReadWarning(UserWarning):
    """
    An image could not be opened/hashed and was skipped from dedup.
    """


@dataclass(frozen=True)
class HashedSample:
    sample: Sample
    phash: imagehash.ImageHash | None  # None if the image failed to open


def compute_phashes(samples: list[Sample], hash_size: int = 8) -> list[HashedSample]:
    """
    Compute a perceptual hash for each sample. Order is preserved.

    Images that fail to open (corrupt file, unreadable format) get phash=None
    and a warning; they are treated as their own singleton cluster downstream
    rather than silently dropped, since silently dropping samples would make
    manifests non-reproducible against the original file list.
    """
    results: list[HashedSample] = []
    for s in samples:
        try:
            with Image.open(s.path) as img:
                img.load()
                h = imagehash.phash(img, hash_size=hash_size)
        except (OSError, UnidentifiedImageError) as e:
            warnings.warn(
                f"{s.sample_id}: could not read image ({e}); treated as unique",
                ImageReadWarning,
                stacklevel=2,
            )
            h = None
        results.append(HashedSample(sample=s, phash=h))
    return results


class _UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, k: str) -> str:
        while self._parent[k] != k:
            self._parent[k] = self._parent[self._parent[k]]
            k = self._parent[k]
        return k

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: smaller id always becomes the new root.
            if rb < ra:
                ra, rb = rb, ra
            self._parent[rb] = ra


def cluster_near_duplicates(
    hashed: list[HashedSample],
    max_distance: int = 8,
) -> dict[str, str]:
    """
    Group samples into clusters by perceptual-hash distance, within each
    class only. Returns {sample_id: cluster_id}, where cluster_id is the
    lexicographically smallest sample_id in that cluster.

    O(n^2) per class. Fine for PlantVillage/PlantDoc/FieldPlant class sizes
    (tens to low thousands per class); revisit with hash-bucket indexing if
    a future dataset has far larger per-class counts.
    """
    by_class: dict[str, list[HashedSample]] = defaultdict(list)
    for h in hashed:
        by_class[h.sample.class_id].append(h)

    assignment: dict[str, str] = {}
    for class_items in by_class.values():
        ids = [h.sample.sample_id for h in class_items]
        uf = _UnionFind(ids)
        n = len(class_items)
        for i in range(n):
            hi = class_items[i].phash
            if hi is None:
                continue
            for j in range(i + 1, n):
                hj = class_items[j].phash
                if hj is None:
                    continue
                if hi - hj <= max_distance:
                    uf.union(class_items[i].sample.sample_id, class_items[j].sample.sample_id)
        for h in class_items:
            assignment[h.sample.sample_id] = uf.find(h.sample.sample_id)

    return assignment


def cluster_sizes(assignment: dict[str, str]) -> dict[str, int]:
    """cluster_id -> number of samples in it. Useful for sanity-checking
    how much near-duplication a dataset actually has."""
    sizes: dict[str, int] = defaultdict(int)
    for cluster_id in assignment.values():
        sizes[cluster_id] += 1
    return dict(sizes)