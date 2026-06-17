import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def add_noise(img, strength, rng):
    if strength <= 0:
        return img
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            delta = rng.randint(-strength, strength)
            pixels[x, y] = (
                max(0, min(255, r + delta)),
                max(0, min(255, g + delta)),
                max(0, min(255, b + delta)),
            )
    return img


def perturb_image(path, out_path, seed, noise, rotate, brightness, contrast, blur):
    rng = random.Random(seed)
    img = Image.open(path).convert("RGB")

    if rotate > 0:
        angle = rng.uniform(-rotate, rotate)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    if brightness > 0:
        factor = 1.0 + rng.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if blur > 0:
        radius = rng.uniform(0, blur)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    img = add_noise(img, noise, rng)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-dir", default="examples")
    parser.add_argument("--outdir", default="examples_noisy")
    parser.add_argument("--parts", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise", type=int, default=10)
    parser.add_argument("--rotate", type=float, default=3.0)
    parser.add_argument("--brightness", type=float, default=0.12)
    parser.add_argument("--contrast", type=float, default=0.12)
    parser.add_argument("--blur", type=float, default=0.5)
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    outdir = Path(args.outdir)
    parts = args.parts or sorted(p.name for p in examples_dir.glob("part_*") if p.is_dir())

    count = 0
    for part in parts:
        for i in range(8):
            src = examples_dir / part / "views" / f"view_{i}.png"
            dst = outdir / part / "views" / f"view_{i}.png"
            if not src.exists():
                raise FileNotFoundError(src)
            perturb_image(
                src,
                dst,
                seed=args.seed + count,
                noise=args.noise,
                rotate=args.rotate,
                brightness=args.brightness,
                contrast=args.contrast,
                blur=args.blur,
            )
            count += 1

    print(f"Created noisy views: {count}")
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
