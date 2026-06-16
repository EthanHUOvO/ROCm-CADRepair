import argparse
import csv
import struct
from pathlib import Path

from PIL import Image


def read_binary_stl_stats(path):
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise RuntimeError("STL is too small")

    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + tri_count * 50
    if expected_size != len(data):
        raise RuntimeError("Only binary STL validation is supported")

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]

    offset = 84
    for _ in range(tri_count):
        offset += 12
        coords = struct.unpack_from("<9f", data, offset)
        for i in range(0, 9, 3):
            for axis in range(3):
                value = coords[i + axis]
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
        offset += 38

    sizes = [maxs[i] - mins[i] for i in range(3)]
    return {
        "triangles": tri_count,
        "bbox_x": sizes[0],
        "bbox_y": sizes[1],
        "bbox_z": sizes[2],
        "stl_ok": tri_count > 0 and all(s > 0 for s in sizes),
    }


def image_stats(path):
    img = Image.open(path).convert("RGB")
    width, height = img.size
    pix = img.load()

    xs = []
    ys = []
    nonwhite = 0
    for y in range(height):
        for x in range(width):
            r, g, b = pix[x, y]
            if min(r, g, b) < 245:
                nonwhite += 1
                xs.append(x)
                ys.append(y)

    total = width * height
    if not xs:
        return {
            "width": width,
            "height": height,
            "nonwhite_ratio": 0.0,
            "margin_left": None,
            "margin_right": None,
            "margin_top": None,
            "margin_bottom": None,
            "view_ok": False,
            "view_issue": "blank",
        }

    left = min(xs)
    right = width - 1 - max(xs)
    top = min(ys)
    bottom = height - 1 - max(ys)
    nonwhite_ratio = nonwhite / total

    issues = []
    if width < 256 or height < 256:
        issues.append("small_image")
    if nonwhite_ratio < 0.01:
        issues.append("object_too_small_or_blank")
    if min(left, right, top, bottom) < 2:
        issues.append("possibly_cropped")
    if nonwhite_ratio > 0.85:
        issues.append("image_overfilled")

    return {
        "width": width,
        "height": height,
        "nonwhite_ratio": nonwhite_ratio,
        "margin_left": left,
        "margin_right": right,
        "margin_top": top,
        "margin_bottom": bottom,
        "view_ok": len(issues) == 0,
        "view_issue": ";".join(issues),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-dir", default="examples")
    parser.add_argument("--out", default="docs/results/cad_view_validation.csv")
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    rows = []

    for part_dir in sorted(examples_dir.glob("part_*")):
        if not part_dir.is_dir():
            continue

        stl_path = part_dir / f"{part_dir.name}_gt.stl"
        stl_stats = {}
        stl_issue = ""
        try:
            stl_stats = read_binary_stl_stats(stl_path)
        except Exception as exc:
            stl_stats = {
                "triangles": None,
                "bbox_x": None,
                "bbox_y": None,
                "bbox_z": None,
                "stl_ok": False,
            }
            stl_issue = str(exc)

        for i in range(8):
            view_path = part_dir / "views" / f"view_{i}.png"
            if view_path.exists():
                stats = image_stats(view_path)
            else:
                stats = {
                    "width": None,
                    "height": None,
                    "nonwhite_ratio": None,
                    "margin_left": None,
                    "margin_right": None,
                    "margin_top": None,
                    "margin_bottom": None,
                    "view_ok": False,
                    "view_issue": "missing",
                }

            rows.append({
                "part": part_dir.name,
                "view": i,
                "stl_path": str(stl_path),
                "view_path": str(view_path),
                "stl_issue": stl_issue,
                **stl_stats,
                **stats,
                "ok": bool(stl_stats.get("stl_ok")) and bool(stats.get("view_ok")),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    failed = [row for row in rows if not row["ok"]]
    print(f"Checked views: {len(rows)}")
    print(f"Failed checks: {len(failed)}")
    if failed:
        for row in failed:
            print(row["part"], "view", row["view"], row["stl_issue"], row["view_issue"])
    print("Saved:", out)


if __name__ == "__main__":
    main()
