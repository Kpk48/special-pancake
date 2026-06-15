"""Small PPM image reader used to keep the prototype dependency-free."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Image:
    width: int
    height: int
    pixels: list[tuple[int, int, int]]


def _read_token(data: bytes, index: int) -> tuple[str, int]:
    while index < len(data):
        char = data[index:index + 1]
        if char == b"#":
            while index < len(data) and data[index:index + 1] not in {b"\n", b"\r"}:
                index += 1
        elif char.isspace():
            index += 1
        else:
            break

    start = index
    while index < len(data) and not data[index:index + 1].isspace():
        index += 1
    return data[start:index].decode("ascii"), index


def load_ppm(path: str | Path) -> Image:
    data = Path(path).read_bytes()
    magic, index = _read_token(data, 0)
    if magic not in {"P3", "P6"}:
        raise ValueError(f"{path} is not a PPM image; expected P3 or P6 header")

    width_text, index = _read_token(data, index)
    height_text, index = _read_token(data, index)
    max_value_text, index = _read_token(data, index)
    width = int(width_text)
    height = int(height_text)
    max_value = int(max_value_text)
    if max_value <= 0 or max_value > 255:
        raise ValueError("Only 8-bit PPM images are supported")

    pixel_count = width * height
    pixels: list[tuple[int, int, int]] = []

    if magic == "P3":
        values: list[int] = []
        while len(values) < pixel_count * 3:
            token, index = _read_token(data, index)
            if not token:
                break
            values.append(int(token))
        if len(values) != pixel_count * 3:
            raise ValueError("PPM file ended before all pixels were read")
        pixels = [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]
    else:
        while index < len(data) and data[index:index + 1].isspace():
            index += 1
        raw = data[index:index + pixel_count * 3]
        if len(raw) != pixel_count * 3:
            raise ValueError("PPM file ended before all pixels were read")
        pixels = [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)]

    return Image(width=width, height=height, pixels=pixels)


def save_ppm(path: str | Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    if len(pixels) != width * height:
        raise ValueError("Pixel count does not match image dimensions")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytes(channel for pixel in pixels for channel in pixel)
    target.write_bytes(header + body)
