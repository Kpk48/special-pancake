#!/usr/bin/env python3
"""Generate an expanded multi-class waste dataset for training.

This script creates synthetic PPM images for additional waste classes to expand
the baseline classifier beyond the original 6 classes.

New classes added:
- textile (clothing, fabric)
- battery (energy waste)
- wood (timber, lumber)
- ceramic (pottery, dishes)
- nylon (plastic films, bags)
"""

import random
from pathlib import Path


def generate_ppm_image(width: int, height: int, seed: int) -> bytes:
    """Generate a valid P6 (binary) PPM image with pseudorandom pixel data."""
    rng = random.Random(seed)
    
    # PPM header (P6 format - binary)
    # Need proper spacing: "P6\n" + width + " " + height + "\n" + maxval + "\n"
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    
    # Generate RGB pixel data as binary
    pixel_data = bytearray()
    for _ in range(width * height):
        pixel_data.append(rng.randint(0, 255))  # R
        pixel_data.append(rng.randint(0, 255))  # G
        pixel_data.append(rng.randint(0, 255))  # B
    
    return header + bytes(pixel_data)


def main() -> None:
    """Generate expanded demo dataset with 11 waste classes."""
    base_dir = Path("data/expanded_waste")
    
    # All classes: original 6 + 5 new ones
    classes = [
        "cardboard",
        "glass",
        "metal",
        "organic",
        "paper",
        "plastic",
        "textile",
        "battery",
        "wood",
        "ceramic",
        "nylon",
    ]
    
    images_per_class = 50  # More images per class for better training
    image_size = 48  # Match demo dataset image size
    
    print(f"Generating expanded waste dataset with {len(classes)} classes...")
    print(f"Generating {images_per_class} images per class ({len(classes) * images_per_class} total)...\n")
    
    for class_name in classes:
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(images_per_class):
            # Generate deterministic but varied seed for each image
            # Use a more stable seed that won't overflow or go negative
            seed = (hash((class_name, i)) & 0x7FFFFFFF)
            img_data = generate_ppm_image(image_size, image_size, seed)
            
            img_path = class_dir / f"{class_name}_{i:03d}.ppm"
            img_path.write_bytes(img_data)
        
        print(f"✓ Created {images_per_class} images in {class_name}/")
    
    print(f"\n✓ Dataset created at {base_dir}/")
    print(f"  Total images: {len(classes) * images_per_class}")
    print(f"  Classes: {', '.join(classes)}")


if __name__ == "__main__":
    main()
