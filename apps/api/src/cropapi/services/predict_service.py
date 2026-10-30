from itertools import count

from cropapi.core.security import check_image_dimensions, check_upload_size, sniff_image_type
from cropapi.schemas.predict import PredictionResponse
from cropapi.services.stub_data import cycle_stub_decision, get_stub_prediction

_call_counter = count()


class PredictService:
    """Stub implementation. Phase 2.6 replaces this with real inference
    behind the same `predict` interface, gated by settings.use_stubs."""

    def __init__(self, max_upload_bytes: int) -> None:
        self._max_upload_bytes = max_upload_bytes

    async def predict(self, image_bytes: bytes) -> PredictionResponse:
        check_upload_size(image_bytes, self._max_upload_bytes)
        sniff_image_type(image_bytes)
        # Real dimension check happens once Pillow is introduced in Phase 2.6;
        # stub trusts the sniffed type and skips pixel decoding.
        check_image_dimensions(width=256, height=256)

        decision = cycle_stub_decision(next(_call_counter))
        return get_stub_prediction(decision)