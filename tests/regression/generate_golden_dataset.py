#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path


WIDTH = 160
HEIGHT = 120


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_generated_files(path: Path, pattern: str) -> None:
    if not path.exists():
        return
    for child in path.glob(pattern):
        if child.is_file():
            child.unlink()


def draw_rotated_rect(center_x: float, center_y: float, width: float, height: float, angle_deg: float) -> list[tuple[int, int, int]]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_w = width / 2.0
    half_h = height / 2.0

    pixels: list[tuple[int, int, int]] = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            dx = x + 0.5 - center_x
            dy = y + 0.5 - center_y
            local_x = dx * cos_a + dy * sin_a
            local_y = -dx * sin_a + dy * cos_a
            if abs(local_x) <= half_w and abs(local_y) <= half_h:
                pixels.append((x, y, 0))
    return pixels


def draw_rotated_ellipse(center_x: float, center_y: float, width: float, height: float, angle_deg: float) -> list[tuple[int, int, int]]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    radius_x = width / 2.0
    radius_y = height / 2.0

    pixels: list[tuple[int, int, int]] = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            dx = x + 0.5 - center_x
            dy = y + 0.5 - center_y
            local_x = dx * cos_a + dy * sin_a
            local_y = -dx * sin_a + dy * cos_a
            value = (local_x * local_x) / (radius_x * radius_x) + (local_y * local_y) / (radius_y * radius_y)
            if value <= 1.0:
                pixels.append((x, y, 0))
    return pixels


def make_image(shape_pixels: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    image = [[(255, 255, 255) for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for x, y, value in shape_pixels:
        image[y][x] = (value, value, value)
    return image


def write_ppm(path: Path, image: list[list[tuple[int, int, int]]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("P3\n")
        handle.write(f"{WIDTH} {HEIGHT}\n")
        handle.write("255\n")
        for row in image:
            handle.write(" ".join(f"{r} {g} {b}" for r, g, b in row))
            handle.write("\n")


def bbox_from_pixels(shape_pixels: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    xs = [x for x, _, _ in shape_pixels]
    ys = [y for _, y, _ in shape_pixels]
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def principal_axis_angle(width: float, height: float, angle_deg: float) -> float:
    angle = angle_deg if width >= height else angle_deg + 90.0
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def shape_pixels(kind: str, cx: float, cy: float, w: float, h: float, angle: float) -> list[tuple[int, int, int]]:
    if kind == "ellipse":
        return draw_rotated_ellipse(cx, cy, w, h, angle)
    return draw_rotated_rect(cx, cy, w, h, angle)


def main() -> None:
    root = Path(__file__).resolve().parent / "golden"
    train_dir = root / "train"
    eval_dir = root / "eval"
    ensure_dir(train_dir)
    ensure_dir(eval_dir)
    clear_generated_files(train_dir, "*.ppm")
    clear_generated_files(eval_dir, "*.ppm")
    clear_generated_files(root, "*.csv")

    train_cases = [
        ("plate_train_a.ppm", "plate", "ellipse", 58.0, 60.0, 74.0, 46.0, 18.0),
        ("plate_train_b.ppm", "plate", "ellipse", 86.0, 68.0, 70.0, 50.0, 28.0),
        ("rod_train_a.ppm", "rod", "rect", 95.0, 60.0, 34.0, 82.0, -18.0),
        ("rod_train_b.ppm", "rod", "rect", 82.0, 56.0, 30.0, 86.0, -12.0),
    ]
    eval_cases = [
        ("plate_eval_a.ppm", "plate", "ellipse", 74.0, 58.0, 72.0, 48.0, 20.0),
        ("plate_eval_b.ppm", "plate", "ellipse", 90.0, 74.0, 68.0, 52.0, 32.0),
        ("rod_eval_a.ppm", "rod", "rect", 58.0, 62.0, 32.0, 84.0, -22.0),
        ("rod_eval_b.ppm", "rod", "rect", 102.0, 54.0, 30.0, 86.0, -14.0),
    ]

    for filename, _label, kind, cx, cy, w, h, angle in train_cases:
        pixels = shape_pixels(kind, cx, cy, w, h, angle)
        write_ppm(train_dir / filename, make_image(pixels))

    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest_file, (
        root / "expected_eval.csv"
    ).open("w", newline="", encoding="utf-8") as expected_file:
        manifest_writer = csv.writer(manifest_file)
        expected_writer = csv.writer(expected_file)
        manifest_writer.writerow(["image", "label"])
        expected_writer.writerow(["image", "label", "centroid_x", "centroid_y", "angle_deg", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])

        for filename, label, kind, cx, cy, w, h, angle in eval_cases:
            pixels = shape_pixels(kind, cx, cy, w, h, angle)
            write_ppm(eval_dir / filename, make_image(pixels))
            bbox_x, bbox_y, bbox_w, bbox_h = bbox_from_pixels(pixels)
            expected_angle = principal_axis_angle(w, h, angle)
            manifest_writer.writerow([f"eval/{filename}", label])
            expected_writer.writerow([f"eval/{filename}", label, cx, cy, expected_angle, bbox_x, bbox_y, bbox_w, bbox_h])


if __name__ == "__main__":
    main()
