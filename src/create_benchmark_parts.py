from pathlib import Path
import math

import cadquery as cq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from render_views_from_stl import render_views


def export_part(result, part_dir: Path, name: str):
    part_dir.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(result, str(part_dir / f"{name}_gt.step"))
    cq.exporters.export(result, str(part_dir / f"{name}_gt.stl"))


def part_001():
    # simple stepped block
    base = cq.Workplane("XY").box(40, 30, 8)
    step1 = cq.Workplane("XY").box(24, 18, 10).translate((0, 0, 9))
    step2 = cq.Workplane("XY").box(12, 10, 8).translate((0, 0, 18))
    return base.union(step1).union(step2)


def part_002():
    # L-shaped bracket
    base = cq.Workplane("XY").box(60, 35, 6)
    wall = cq.Workplane("XY").box(60, 6, 35).translate((0, 14.5, 17.5))
    rib1 = cq.Workplane("XY").box(8, 20, 25).translate((-18, 5, 12.5))
    rib2 = cq.Workplane("XY").box(8, 20, 25).translate((18, 5, 12.5))
    result = base.union(wall).union(rib1).union(rib2)
    result = result.faces(">Z").workplane().pushPoints([(-18, 0), (18, 0)]).hole(5)
    return result


def part_003():
    # multi-hole flange bracket
    base = cq.Workplane("XY").circle(28).extrude(6)
    base = base.faces(">Z").workplane().hole(10)
    pts = []
    for a in [0, 90, 180, 270]:
        r = math.radians(a)
        pts.append((18 * math.cos(r), 18 * math.sin(r)))
    base = base.faces(">Z").workplane().pushPoints(pts).hole(4)

    block = cq.Workplane("XY").box(24, 14, 28).translate((0, 0, 20))
    boss = cq.Workplane("XY").circle(8).extrude(10).translate((0, 0, 34))
    rib1 = cq.Workplane("XY").box(6, 20, 22).translate((-15, 0, 17))
    rib2 = cq.Workplane("XY").box(6, 20, 22).translate((15, 0, 17))
    return base.union(block).union(boss).union(rib1).union(rib2)


def part_004():
    # rectangular connector with two holes and top boss
    result = cq.Workplane("XY").box(55, 32, 10)
    result = result.faces(">Z").workplane().pushPoints([(-16, 0), (16, 0)]).hole(6)
    boss = cq.Workplane("XY").box(22, 14, 14).translate((0, 0, 12))
    return result.union(boss)


def part_005():
    # dual cylinder posts on base
    base = cq.Workplane("XY").box(60, 30, 8)
    post1 = cq.Workplane("XY").circle(7).extrude(24).translate((-17, 0, 4))
    post2 = cq.Workplane("XY").circle(7).extrude(24).translate((17, 0, 4))
    result = base.union(post1).union(post2)
    result = result.faces(">Z").workplane().pushPoints([(-17, 0), (17, 0)]).hole(4)
    return result


def part_006():
    # T-shaped support
    base = cq.Workplane("XY").box(64, 22, 8)
    stem = cq.Workplane("XY").box(16, 50, 14).translate((0, 0, 11))
    top = cq.Workplane("XY").box(44, 16, 14).translate((0, 18, 22))
    return base.union(stem).union(top)


def part_007():
    # U-shaped clamp
    base = cq.Workplane("XY").box(50, 40, 8)
    left = cq.Workplane("XY").box(10, 40, 32).translate((-20, 0, 16))
    right = cq.Workplane("XY").box(10, 40, 32).translate((20, 0, 16))
    result = base.union(left).union(right)
    result = result.faces(">Z").workplane().pushPoints([(-20, 0), (20, 0)]).hole(5)
    return result


def part_008():
    # multi-boss mechanical base
    base = cq.Workplane("XY").box(72, 42, 8)
    boss1 = cq.Workplane("XY").box(18, 18, 18).translate((-24, 0, 13))
    boss2 = cq.Workplane("XY").circle(9).extrude(20).translate((0, 0, 8))
    boss3 = cq.Workplane("XY").box(18, 18, 14).translate((24, 0, 11))
    result = base.union(boss1).union(boss2).union(boss3)
    result = result.faces(">Z").workplane().pushPoints([(-24, 0), (0, 0), (24, 0)]).hole(4)
    return result


PARTS = {
    "part_001": part_001,
    "part_002": part_002,
    "part_003": part_003,
    "part_004": part_004,
    "part_005": part_005,
    "part_006": part_006,
    "part_007": part_007,
    "part_008": part_008,
}


def cuboid_faces(origin, size):
    x, y, z = origin
    dx, dy, dz = size
    p = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    return [
        [p[0], p[1], p[2], p[3]],
        [p[4], p[5], p[6], p[7]],
        [p[0], p[1], p[5], p[4]],
        [p[2], p[3], p[7], p[6]],
        [p[1], p[2], p[6], p[5]],
        [p[0], p[3], p[7], p[4]],
    ]


def cylinder_faces(cx, cy, z0, radius, height, segments=48):
    bottom = []
    top = []
    for i in range(segments):
        a = 2 * math.pi * i / segments
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        bottom.append((x, y, z0))
        top.append((x, y, z0 + height))

    faces = [bottom, top]
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([bottom[i], bottom[j], top[j], top[i]])
    return faces


def add_poly(ax, faces, color=(0.72, 0.72, 0.72, 1.0), edge=(0.1, 0.1, 0.1, 1.0), lw=0.5):
    poly = Poly3DCollection(
        faces,
        facecolor=color,
        edgecolor=edge,
        linewidths=lw,
        alpha=1.0,
    )
    ax.add_collection3d(poly)


def render_schematic(part_name: str, outdir: Path):
    """
    用 matplotlib 生成统一风格的 8 视角图。
    这里是用于 Zero-to-CAD 输入的 clean rendered views，不追求精确布尔渲染，
    但会展示每个 part 的主要工程结构。
    """
    view_dir = outdir / "views"
    view_dir.mkdir(parents=True, exist_ok=True)

    def add_scene(ax):
        if part_name == "part_001":
            add_poly(ax, cuboid_faces((-20, -15, 0), (40, 30, 8)))
            add_poly(ax, cuboid_faces((-12, -9, 8), (24, 18, 10)))
            add_poly(ax, cuboid_faces((-6, -5, 18), (12, 10, 8)))

        elif part_name == "part_002":
            add_poly(ax, cuboid_faces((-30, -17.5, 0), (60, 35, 6)))
            add_poly(ax, cuboid_faces((-30, 11.5, 6), (60, 6, 35)))
            add_poly(ax, cuboid_faces((-22, -6, 6), (8, 20, 25)))
            add_poly(ax, cuboid_faces((14, -6, 6), (8, 20, 25)))
            for x in [-18, 18]:
                add_poly(ax, cylinder_faces(x, 0, 6.2, 3, 0.2), color=(0.02, 0.02, 0.02, 1.0))

        elif part_name == "part_003":
            add_poly(ax, cylinder_faces(0, 0, 0, 28, 6))
            add_poly(ax, cylinder_faces(0, 0, 6.2, 5, 0.2), color=(0.02, 0.02, 0.02, 1.0))
            for a in [0, 90, 180, 270]:
                r = math.radians(a)
                add_poly(ax, cylinder_faces(18 * math.cos(r), 18 * math.sin(r), 6.3, 2.2, 0.2), color=(0.02, 0.02, 0.02, 1.0))
            add_poly(ax, cuboid_faces((-12, -7, 6), (24, 14, 28)))
            add_poly(ax, cylinder_faces(0, 0, 34, 8, 10))
            add_poly(ax, cuboid_faces((-18, -10, 6), (6, 20, 22)))
            add_poly(ax, cuboid_faces((12, -10, 6), (6, 20, 22)))

        elif part_name == "part_004":
            add_poly(ax, cuboid_faces((-27.5, -16, 0), (55, 32, 10)))
            add_poly(ax, cuboid_faces((-11, -7, 10), (22, 14, 14)))
            for x in [-16, 16]:
                add_poly(ax, cylinder_faces(x, 0, 10.2, 3, 0.2), color=(0.02, 0.02, 0.02, 1.0))

        elif part_name == "part_005":
            add_poly(ax, cuboid_faces((-30, -15, 0), (60, 30, 8)))
            for x in [-17, 17]:
                add_poly(ax, cylinder_faces(x, 0, 8, 7, 24))
                add_poly(ax, cylinder_faces(x, 0, 32.2, 2.2, 0.2), color=(0.02, 0.02, 0.02, 1.0))

        elif part_name == "part_006":
            add_poly(ax, cuboid_faces((-32, -11, 0), (64, 22, 8)))
            add_poly(ax, cuboid_faces((-8, -25, 8), (16, 50, 14)))
            add_poly(ax, cuboid_faces((-22, 10, 22), (44, 16, 14)))

        elif part_name == "part_007":
            add_poly(ax, cuboid_faces((-25, -20, 0), (50, 40, 8)))
            add_poly(ax, cuboid_faces((-25, -20, 8), (10, 40, 32)))
            add_poly(ax, cuboid_faces((15, -20, 8), (10, 40, 32)))
            for x in [-20, 20]:
                add_poly(ax, cylinder_faces(x, 0, 8.2, 2.5, 0.2), color=(0.02, 0.02, 0.02, 1.0))

        elif part_name == "part_008":
            add_poly(ax, cuboid_faces((-36, -21, 0), (72, 42, 8)))
            add_poly(ax, cuboid_faces((-33, -9, 8), (18, 18, 18)))
            add_poly(ax, cylinder_faces(0, 0, 8, 9, 20))
            add_poly(ax, cuboid_faces((15, -9, 8), (18, 18, 14)))
            for x in [-24, 0, 24]:
                add_poly(ax, cylinder_faces(x, 0, 28.3, 2.2, 0.2), color=(0.02, 0.02, 0.02, 1.0))

    views = [
        (20, 0),
        (20, 180),
        (20, 90),
        (20, -90),
        (90, 0),
        (-60, 0),
        (30, 45),
        (30, 135),
    ]

    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        add_scene(ax)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-45, 45)
        ax.set_ylim(-45, 45)
        ax.set_zlim(0, 55)
        ax.set_box_aspect((1, 1, 0.8))
        ax.axis("off")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        plt.tight_layout(pad=0)
        plt.savefig(view_dir / f"view_{i}.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main():
    root = Path("./examples")
    root.mkdir(parents=True, exist_ok=True)

    for name, fn in PARTS.items():
        print("Creating", name)
        part_dir = root / name
        result = fn()
        export_part(result, part_dir, name)
        render_views(part_dir / f"{name}_gt.stl", part_dir / "views")
        print("Saved:", part_dir)

    print("All benchmark parts created.")


if __name__ == "__main__":
    main()
