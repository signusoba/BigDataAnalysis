"""Shared input preparation for training and inference. White=0, ink=1."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 64


def _otsu_threshold(values: np.ndarray) -> int:
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    weight = np.cumsum(histogram)
    mean = np.cumsum(histogram * np.arange(256))
    denominator = weight * (weight[-1] - weight)
    score = np.zeros(256)
    valid = denominator > 0
    score[valid] = (mean[-1] * weight[valid] - mean[valid] * weight[-1]) ** 2 / denominator[valid]
    return int(np.argmax(score))


def preprocess_image(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Return float32 (1, size, size), retaining detached dots and aspect ratio.

    Expects one isolated glyph on a mostly uniform background. Detects light or
    dark backgrounds from the image border; never uses only the largest component.
    """
    if not isinstance(image, Image.Image) or image.width < 3 or image.height < 3:
        raise ValueError("Please provide a readable image containing one isolated Persian letter.")
    image = ImageOps.exif_transpose(image)
    if image.mode == "RGBA" or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.size > 16_000_000:
        raise ValueError("Image is too large; use a crop smaller than 16 megapixels.")
    if float(gray.max()) - float(gray.min()) < 20:
        raise ValueError("Image is blank or too low contrast to recognize.")
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    ink = 255 - gray if np.median(border) >= 128 else gray.copy()
    threshold = _otsu_threshold(ink)
    mask = ink > max(threshold, 16)
    rows, cols = np.nonzero(mask)
    if len(rows) < 4 or len(rows) / gray.size > 0.95:
        raise ValueError("No isolated letter was detected; use a contrasting plain background.")
    # All ink components participate, including diacritic dots.
    crop = Image.fromarray(ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1])
    inset = max(4, round(size * 0.125))
    available = size - 2 * inset
    scale = min(available / crop.width, available / crop.height)
    shape = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(shape, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return np.asarray(canvas, dtype=np.float32)[None, :, :] / 255.0


def preview_data_url(array: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(1 - array.squeeze(), 0, 1) * 255)).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
