from __future__ import annotations

from collections.abc import Iterator

from cropml.data.loaders.base import BaseLoader, LoaderConfigError, RawItem


class FieldPlantLoader(BaseLoader):
    def _discover(self) -> Iterator[RawItem]:
        cfg = self.config
        if cfg.layout != "class_folders":
            raise LoaderConfigError(f"fieldplant supports layout 'class_folders', got '{cfg.layout}'")

        annotation_format = cfg.options.get("annotation_format", "none")
        if annotation_format != "none":
            raise LoaderConfigError(
                f"annotation_format '{annotation_format}' is not supported yet. The real "
                "FieldPlant folder structure is unverified; send it and this loader "
                "will be extended."
            )

        yield from self._scan_class_folders(cfg.root_dir, split=None)

