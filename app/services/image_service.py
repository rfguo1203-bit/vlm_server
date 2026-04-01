from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException, status

from app.core.config import Settings


def _load_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pillow is required for image inputs. Please install `pillow` in the server environment.",
        ) from exc
    return Image


def validate_image_count(image_count: int, settings: Settings) -> None:
    if image_count > settings.max_input_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many images in request: {image_count}. "
                f"Maximum allowed is {settings.max_input_images}."
            ),
        )


def load_image_from_path(path_str: str, settings: Settings):
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image file does not exist: {path}",
        )

    file_size = path.stat().st_size
    if file_size > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image file is too large: {file_size} bytes. "
                f"Maximum allowed is {settings.max_image_bytes} bytes."
            ),
        )

    Image = _load_pillow_image()
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to open image file `{path}`: {exc}",
        ) from exc


def load_image_from_base64(data: str, settings: Settings):
    try:
        raw_bytes = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image payload: {exc}",
        ) from exc

    byte_count = len(raw_bytes)
    if byte_count > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Base64 image is too large after decoding: {byte_count} bytes. "
                f"Maximum allowed is {settings.max_image_bytes} bytes."
            ),
        )

    Image = _load_pillow_image()
    try:
        with Image.open(BytesIO(raw_bytes)) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image from base64 payload: {exc}",
        ) from exc


def load_image_from_data_url(url: str, settings: Settings):
    prefix = "base64,"
    if prefix not in url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only base64-encoded data URLs are supported.",
        )
    encoded = url.split(prefix, 1)[1]
    return load_image_from_base64(encoded, settings)


def normalize_local_image_reference(url: str, settings: Settings):
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return load_image_from_path(parsed.path, settings)
    if parsed.scheme == "data":
        return load_image_from_data_url(url, settings)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Remote image URLs are not supported yet. Please use local path or base64 input.",
    )
