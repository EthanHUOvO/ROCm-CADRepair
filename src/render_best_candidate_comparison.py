import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render_views_from_stl import render_views


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def paste_contain(canvas, source, box):
    x0, y0, x1, y1 = box
    image = source.copy()
    image.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    canvas.paste(image, (x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-dir", required=True)
    parser.add_argument("--target-views", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-faces", type=int, default=12000)
    args = parser.parse_args()

    part_dir = Path(args.part_dir)
    best = json.loads((part_dir / "best_candidate.json").read_text(encoding="utf-8"))
    candidate = best.get("candidate")
    if candidate is None:
        candidate = int(Path(best["candidate_stl"]).stem.split("_")[-1])
    candidate = int(candidate)
    stl = part_dir / f"candidate_{candidate}.stl"
    if not stl.exists():
        raise FileNotFoundError(stl)

    tile_w, tile_h = 400, 330
    canvas = Image.new("RGB", (tile_w * 4, tile_h * 2 + 82), (244, 248, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 20),
        f"Eight-view audit | selected candidate {candidate}",
        font=load_font(28, True),
        fill=(20, 59, 105),
    )
    draw.text((910, 27), "TARGET", font=load_font(20, True), fill=(20, 132, 126))
    draw.text((1080, 27), "SELECTED", font=load_font(20, True), fill=(221, 106, 45))

    with tempfile.TemporaryDirectory() as temp_dir:
        render_views(
            stl,
            temp_dir,
            image_size=args.image_size,
            max_faces=args.max_faces,
        )
        for index in range(8):
            row, col = divmod(index, 4)
            x = col * tile_w
            y = 82 + row * tile_h
            draw.rounded_rectangle(
                (x + 10, y + 10, x + tile_w - 10, y + tile_h - 10),
                radius=12,
                fill=(255, 255, 255),
                outline=(207, 220, 230),
                width=2,
            )
            target = Image.open(Path(args.target_views) / f"view_{index}.png").convert("RGB")
            selected = Image.open(Path(temp_dir) / f"view_{index}.png").convert("RGB")
            paste_contain(canvas, target, (x + 22, y + 48, x + 195, y + 292))
            paste_contain(canvas, selected, (x + 205, y + 48, x + 378, y + 292))
            draw.text((x + 28, y + 20), f"View {index + 1}", font=load_font(18, True), fill=(36, 52, 66))
            draw.text((x + 72, y + 295), "Target", font=load_font(15), fill=(20, 132, 126))
            draw.text((x + 255, y + 295), "Selected", font=load_font(15), fill=(221, 106, 45))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
