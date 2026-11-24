import argparse
import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FILENAME_RE = re.compile(r"^(?P<writer>[^_]+)_(?P<label>[^_]+)_(?P<sheet>sheet\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_largest_quad_contour(binary: np.ndarray):
    # Find contours and approximate polygons; return biggest 4-point polygon.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50_000:  # reject tiny
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx.reshape(4, 2)
            best_area = area

    return best, best_area


def warp_to_square(img_gray: np.ndarray, quad: np.ndarray, size: int = 1000) -> np.ndarray:
    quad = order_points(quad.astype(np.float32))
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_gray, M, (size, size))
    return warped


def slice_grid(warped: np.ndarray, grid_n: int = 10, pad: int = 6, out_size: int = 28):
    # warped is square image containing the full grid area which is sliced it into grid_n x grid_n cells
    # pad is pixels cropped from each cell border to remove grid lines

    h, w = warped.shape[:2]
    cell_h = h // grid_n
    cell_w = w // grid_n

    cells = []
    for r in range(grid_n):
        for c in range(grid_n):
            y0 = r * cell_h
            x0 = c * cell_w
            cell = warped[y0:y0 + cell_h, x0:x0 + cell_w]

            # remove border/grid lines
            cell = cell[pad:cell.shape[0] - pad, pad:cell.shape[1] - pad]

            # normalize to 0..255 and resize
            cell = cv2.resize(cell, (out_size, out_size), interpolation=cv2.INTER_AREA)
            cells.append(cell)

    return cells


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def process_sheet(path: Path, out_dir: Path, grid_n: int, out_size: int, debug_dir: Path | None):
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Bad filename: {path.name}. Expected writer_label_sheet#.jpg (e.g., s12A_3_sheet1.jpg)")

    writer = m.group("writer")
    label = m.group("label")
    sheet_id = m.group("sheet")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess for contour detection: strong edges of grid / border
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Thicken lines to help contour detection
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    quad, area = find_largest_quad_contour(edges)
    if quad is None:
        raise RuntimeError(
            f"Could not find grid/border quad in {path.name}. "
            f"Tip: use document scan mode, ensure full page is visible, avoid shadows."
        )
    
    warped = warp_to_square(gray, quad, size=1200)

    # Improve contrast 
    warped = cv2.equalizeHist(warped)

    if debug_dir is not None:
        ensure_dir(debug_dir)
        cv2.imwrite(str(debug_dir / f"{path.stem}__warped.png"), warped)

    cells = slice_grid(warped, grid_n=grid_n, pad=7, out_size=out_size)

    # Save cells
    label_dir = out_dir / "images" / str(label)
    ensure_dir(label_dir)

    rows = []
    for idx, cell in enumerate(cells):
        fname = f"{writer}_{label}_{sheet_id}_{idx:03d}.png"
        fpath = label_dir / fname
        cv2.imwrite(str(fpath), cell)
        rows.append({
            "filepath": str(fpath.as_posix()),
            "label": str(label),
            "writer_id": str(writer),
            "sheet_id": str(sheet_id),
            "source_sheet": str(path.name),
            "cell_index": idx
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder containing scanned sheet images (jpg/png)")
    ap.add_argument("--out", required=True, help="Output dataset folder")
    ap.add_argument("--grid", type=int, default=10, help="Grid size (default 10 for 10x10)")
    ap.add_argument("--size", type=int, default=28, help="Output cell image size (default 28)")
    ap.add_argument("--debug", action="store_true", help="Save warped grid previews for troubleshooting")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    debug_dir = out_dir / "debug" if args.debug else None

    all_rows = []
    for p in sorted(in_dir.iterdir()):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        rows = process_sheet(p, out_dir, grid_n=args.grid, out_size=args.size, debug_dir=debug_dir)
        all_rows.extend(rows)

    labels_path = out_dir / "labels.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(labels_path, index=False)

    # Summary
    counts = df.groupby("label").size().sort_index()
    print("\nSaved dataset to:", out_dir.resolve())
    print("labels.csv:", labels_path.resolve())
    print("\nCounts per label:")
    print(counts.to_string())
    print("\nRun with --debug if crops look wrong.\n")


if __name__ == "__main__":
    main()
