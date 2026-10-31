import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cropml.data.label_space import ClassTag
from cropml.data.loaders.base import Sample


@pytest.fixture
def tmp_data_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    A temp dir, with CROPML_DATA_ROOT pointed at it for the test's duration.
    """
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setenv("CROPML_DATA_ROOT", str(tmp))
    return tmp


@pytest.fixture
def make_sample() -> Callable[..., Sample]:
    """
    Factory fixture: returns a function for building a Sample with
    sensible defaults, for tests that don't need a real file.
    """

    def _make(
        sample_id: str,
        class_id: str,
        tag: ClassTag = ClassTag.IN_DOMAIN,
        dataset: str = "ds",
        domain: str = "lab",
        split: str | None = None,
        path: Path | None = None,
    ) -> Sample:
        return Sample(
            sample_id=sample_id,
            path=path or Path(sample_id),
            dataset=dataset,
            domain=domain,
            class_id=class_id,
            tag=tag,
            split=split,
        )

    return _make


@pytest.fixture
def make_image_file() -> Callable[..., None]:
    """
    Factory fixture: returns a function that writes a reproducible-but
    -non-trivial image, so phash has real structure to hash (solid colors
    defeat phash - see step 5's lesson).
    """

    def _make(path: Path, seed: int, size: tuple[int, int] = (48, 48)) -> None:
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)

    return _make