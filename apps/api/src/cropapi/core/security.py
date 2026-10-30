import imghdr

from cropapi.core.errors import AppError

ALLOWED_IMAGE_TYPES = {"jpeg", "png"}
MIN_DIMENSION_PX = 32
MAX_DIMENSION_PX = 8000  # guards against decompression-bomb style images


def sniff_image_type(data: bytes) -> str:
    """
    Detect actual image type from content, not from the filename/extension.
    """
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
    """
    Rejects implausible dimensions that could indicate a decompression bomb.
    """
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
    





