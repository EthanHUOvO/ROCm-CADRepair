import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


VIEWS = [
    (20, 0),
    (20, 180),
    (20, 90),
    (20, -90),
    (90, 0),
    (-60, 0),
    (30, 45),
    (30, 135),
]


def read_binary_stl(path):
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise RuntimeError(f"Invalid STL file: {path}")

    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + tri_count * 50
    if expected_size != len(data):
        raise ValueError("Not a binary STL with the expected byte length.")

    faces = np.empty((tri_count, 3, 3), dtype=np.float32)
    offset = 84
    for i in range(tri_count):
        # normal: 12 bytes, vertices: 36 bytes, attribute byte count: 2 bytes
        offset += 12
        faces[i] = np.array(struct.unpack_from("<9f", data, offset), dtype=np.float32).reshape(3, 3)
        offset += 38
    return faces


def read_ascii_stl(path):
    vertices = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(vertices) % 3 != 0 or not vertices:
        raise RuntimeError(f"Invalid ASCII STL file: {path}")

    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3, 3)


def read_stl(path):
    try:
        return read_binary_stl(path)
    except Exception:
        return read_ascii_stl(path)


def axis_limits(points, margin_ratio=0.12):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max() / 2)
    radius = max(radius, 1.0) * (1 + margin_ratio)
    return [(center[i] - radius, center[i] + radius) for i in range(3)]


def render_views(stl_path, outdir, image_size=512, dpi=100, line_width=0.22):
    faces = read_stl(stl_path)
    points = faces.reshape(-1, 3)
    limits = axis_limits(points)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, (elev, azim) in enumerate(VIEWS):
        fig = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        mesh = Poly3DCollection(
            faces,
            facecolor=(0.72, 0.72, 0.72, 1.0),
            edgecolor=(0.18, 0.18, 0.18, 1.0),
            linewidths=line_width,
            alpha=1.0,
        )
        ax.add_collection3d(mesh)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.axis("off")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(outdir / f"view_{i}.png", dpi=dpi, facecolor="white", pad_inches=0)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    render_views(args.stl, args.outdir, image_size=args.image_size, dpi=args.dpi)
    print("Rendered views:", args.outdir)


if __name__ == "__main__":
    main()
