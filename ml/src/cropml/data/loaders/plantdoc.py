from __future__ import annotations

from collections.abc import Iterator

from cropml.data.loaders.base import BaseLoader, LoaderConfigError, RawItem


class PlantDocLoader(BaseLoader):
    def _discover(self) -> Iterator[RawItem]:
        cfg = self.config
        if cfg.layout != "predefined_splits":
            raise LoaderConfigError(f"plantdoc supports layout 'predefined_splits', got '{cfg.layout}'")
        if not cfg.split_dirs:
            raise LoaderConfigError("plantdoc needs split_dirs (e.g. train/test) in its config")

        # Samples keep the official split name. Per the config, the official
        # test split is target-domain evaluation only; the splitter enforces it.
        for split, dirname in cfg.split_dirs.items():
            base = self._find_subdir_ci(cfg.root_dir, dirname)
            yield from self._scan_class_folders(base, split=split)

