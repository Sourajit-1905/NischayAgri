import imghdr
import io

from PIL import Image, ImageOps, UnidentifiedImageError

from cropapi.core.errors import AppError

ALLOWED_IMAGE_TYPES = {"jpeg", "png"}
MIN_DIMENSION_PX = 32
MAX_DIMENSION_PX = 8000
# Decompression bomb: PIL raises DecompressionBombError above this by default
# (128MP). We set a tighter limit appropriate for leaf photos.
MAX_PIXELS = 20_000_000  # 20 MP


def sniff_image_type(data: bytes) -> str:
    kind = imghdr.what(None, h=data)
    if kind not in ALLOWED_IMAGE_TYPES:
        raise AppError(
            code="unsupported_image_type",
            message="Uploaded file is not a JPEG or PNG image",
            status_code=400,
        )
    return kind


def check_upload_size(data: bytes, max_bytes: int) -> None:
    if len(data) == 0:
        raise AppError(code="empty_upload", message="Uploaded file is empty", status_code=400)
    if len(data) > max_bytes:
        raise AppError(
            code="upload_too_large",
            message=f"Upload exceeds the {max_bytes} byte limit",
            status_code=413,
        )


def check_image_dimensions(width: int, height: int) -> None:
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        raise AppError(
            code="image_too_small",
            message=f"Image must be at least {MIN_DIMENSION_PX}px on each side",
            status_code=400,
        )
    if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
        raise AppError(
            code="image_too_large",
            message=f"Image must be at most {MAX_DIMENSION_PX}px on each side",
            status_code=400,
        )


def strip_exif_and_validate(data: bytes) -> bytes:
    """Opens the image with PIL, strips EXIF metadata, enforces the
    decompression-bomb pixel limit, and returns clean JPEG bytes.

    Decompression-bomb protection: PIL raises DecompressionBombError
    for images exceeding Image.MAX_IMAGE_PIXELS. We set a tighter limit
    before opening, then restore it afterward."""
    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Image.DecompressionBombError:
            raise AppError(
                code="decompression_bomb",
                message=f"Image exceeds the {MAX_PIXELS // 1_000_000}MP pixel limit",
                status_code=400,
            )
        except UnidentifiedImageError:
            raise AppError(
                code="unreadable_image",
                message="Image file could not be decoded",
                status_code=400,
            )

        # Strip EXIF by re-encoding as JPEG without metadata.
        img_no_exif = ImageOps.exif_transpose(img).convert("RGB")
        buf = io.BytesIO()
        img_no_exif.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit