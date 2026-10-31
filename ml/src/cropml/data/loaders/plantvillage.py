from __future__ import annotations

from collections.abc import Iterator

from cropml.data.loaders.base import BaseLoader, LoaderConfigError, RawItem


class PlantVillageLoader(BaseLoader):
    def _discover(self) -> Iterator[RawItem]:
        cfg = self.config
        if cfg.layout != "class_folders":
            raise LoaderConfigError(f"plantvillage supports layout 'class_folders', got '{cfg.layout}'")

        base = cfg.root_dir
        if cfg.variant_subdir:
            base = self._find_subdir_ci(base, cfg.variant_subdir)

        # Heuristic only: it catches folders named like "...augmented...".
        if not cfg.options.get("allow_offline_augmented", False):
            for p in {cfg.root_dir, base}:
                if "augment" in p.name.lower():
                    raise LoaderConfigError(
                        f"'{p.name}' looks like an offline-augmented copy of PlantVillage. "
                        "That bakes near-duplicates into every split. Use the original "
                        "images, or set allow_offline_augmented: true if you are sure."
                    )

        yield from self._scan_class_folders(base, split=None)