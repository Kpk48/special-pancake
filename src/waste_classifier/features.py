"""Feature extraction for waste images."""

from __future__ import annotations

from pathlib import Path

from .image_io import Image, load_ppm


def _channel_stats(values: list[int]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean / 255.0, (variance ** 0.5) / 255.0


def extract_features_from_image(image: Image) -> list[float]:
    red = [pixel[0] for pixel in image.pixels]
    green = [pixel[1] for pixel in image.pixels]
    blue = [pixel[2] for pixel in image.pixels]

    r_mean, r_std = _channel_stats(red)
    g_mean, g_std = _channel_stats(green)
    b_mean, b_std = _channel_stats(blue)

    brightness = [(r + g + b) / 3 for r, g, b in image.pixels]
    bright_mean, bright_std = _channel_stats([int(value) for value in brightness])

    edge_total = 0.0
    comparisons = 0
    for y in range(image.height):
        row = y * image.width
        for x in range(image.width - 1):
            left = brightness[row + x]
            right = brightness[row + x + 1]
            edge_total += abs(left - right) / 255.0
            comparisons += 1
    texture = edge_total / comparisons if comparisons else 0.0

    warm_ratio = sum(1 for r, g, b in image.pixels if r > g and r > b) / len(image.pixels)
    green_ratio = sum(1 for r, g, b in image.pixels if g > r and g > b) / len(image.pixels)
    blue_ratio = sum(1 for r, g, b in image.pixels if b > r and b > g) / len(image.pixels)

    return [
        r_mean,
        g_mean,
        b_mean,
        r_std,
        g_std,
        b_std,
        bright_mean,
        bright_std,
        texture,
        warm_ratio,
        green_ratio,
        blue_ratio,
    ]


def extract_features(path: str | Path) -> list[float]:
    return extract_features_from_image(load_ppm(path))
