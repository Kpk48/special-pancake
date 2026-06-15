"""Generate a small synthetic waste image dataset in PPM format."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waste_classifier.image_io import save_ppm


ROOT = Path("data/demo_waste")
WIDTH = 48
HEIGHT = 48
IMAGES_PER_CLASS = 32

CLASS_STYLES = {
    "cardboard": {"base": (170, 120, 62), "noise": 28, "stripe": (120, 82, 42)},
    "glass": {"base": (80, 170, 190), "noise": 34, "stripe": (210, 240, 245)},
    "metal": {"base": (150, 155, 160), "noise": 22, "stripe": (225, 225, 220)},
    "organic": {"base": (70, 135, 58), "noise": 40, "stripe": (115, 72, 36)},
    "paper": {"base": (220, 214, 190), "noise": 18, "stripe": (170, 165, 150)},
    "plastic": {"base": (55, 95, 210), "noise": 36, "stripe": (230, 230, 245)},
}


def clamp(value: int) -> int:
    return max(0, min(255, value))


def pixel_for(style: dict[str, object], x: int, y: int, rng: random.Random) -> tuple[int, int, int]:
    base = style["base"]
    stripe = style["stripe"]
    noise = int(style["noise"])
    assert isinstance(base, tuple)
    assert isinstance(stripe, tuple)

    use_stripe = (x + y) % 13 == 0 or (x // 8) % 2 == 0 and y % 11 == 0
    color = stripe if use_stripe else base
    jitter = rng.randint(-noise, noise)
    return tuple(clamp(int(channel) + jitter + rng.randint(-8, 8)) for channel in color)


def main() -> None:
    for label, style in CLASS_STYLES.items():
        for index in range(IMAGES_PER_CLASS):
            rng = random.Random(f"{label}-{index}")
            pixels = [
                pixel_for(style, x, y, rng)
                for y in range(HEIGHT)
                for x in range(WIDTH)
            ]
            save_ppm(ROOT / label / f"{label}_{index:03d}.ppm", WIDTH, HEIGHT, pixels)
    print(f"Generated {len(CLASS_STYLES) * IMAGES_PER_CLASS} demo images in {ROOT}")


if __name__ == "__main__":
    main()
