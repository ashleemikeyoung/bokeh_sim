#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
#
# compare_depth_models.py: line up the depth maps and renders that bokeh_sim made with
# different depth models, so they can be judged side by side.
# Copyright (C) 2026 Qavats Ranch, LLC
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.
"""
Usage:
    python compare_depth_models.py _DSF5088.jpeg da2=out-da2 depthpro=out-depthpro moge2=out-moge2 moge3=out-moge3

Each folder must hold the files bokeh_sim wrote with --save-depth: <photo>_depth.png and one render.
Writes compare_depth.png (the depth maps) and compare_crops.png (zoomed crops of the original and
every render). Add --crop name:x0,y0,x1,y1 (fractions of the photo) to choose your own crops.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_CROPS = {
    "whiskers": (0.30, 0.37, 0.49, 0.52),
    "ear edge": (0.62, 0.12, 0.75, 0.33),
    "poster edge": (0.04, 0.16, 0.22, 0.42),
}
TILE_W = 520


def label(im, text):
    out = im.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, 8 * len(text) + 12, 20), fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255))
    return out


def find_render(folder, stem):
    cands = sorted(p for p in Path(folder).glob(f"{stem}_*.jpg") if "_depth" not in p.name)
    if not cands:
        sys.exit(f"No render named {stem}_*.jpg found in {folder}")
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="Compare bokeh_sim results from different depth models.")
    ap.add_argument("photo")
    ap.add_argument("runs", nargs="+", help="label=folder, for example moge3=out-moge3")
    ap.add_argument("--crop", action="append", default=[], help="name:x0,y0,x1,y1 as fractions")
    args = ap.parse_args()

    photo = Path(args.photo)
    stem = photo.stem
    original = Image.open(photo).convert("RGB")
    runs = []
    for r in args.runs:
        name, _, folder = r.partition("=")
        if not folder:
            sys.exit(f"Give each run as label=folder, got {r!r}")
        depth = Path(folder) / f"{stem}_depth.png"
        if not depth.exists():
            sys.exit(f"Missing {depth}. Run bokeh_sim with --save-depth --out-dir {folder}")
        runs.append((name, Image.open(depth).convert("L"), Image.open(find_render(folder, stem)).convert("RGB")))

    # depth maps side by side
    h = int(TILE_W * original.height / original.width)
    tiles = [label(im.resize((TILE_W, h)), name) for name, im, _ in runs]
    sheet = Image.new("RGB", (TILE_W * len(tiles), h))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * TILE_W, 0))
    sheet.save("compare_depth.png")
    print("saved compare_depth.png")

    # zoomed crops: one row per crop, original first, then each model's render
    crops = dict(DEFAULT_CROPS)
    for c in args.crop:
        name, _, box = c.partition(":")
        crops[name] = tuple(float(v) for v in box.split(","))
    rows = []
    for cname, (x0, y0, x1, y1) in crops.items():
        row = []
        for tag, im in [("original", original)] + [(n, r) for n, _, r in runs]:
            W, H = im.size
            tile = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
            th = int(TILE_W * tile.height / tile.width)
            row.append(label(tile.resize((TILE_W, th), Image.LANCZOS), f"{cname}: {tag}"))
        rh = max(t.height for t in row)
        strip = Image.new("RGB", (TILE_W * len(row), rh))
        for i, t in enumerate(row):
            strip.paste(t, (i * TILE_W, 0))
        rows.append(strip)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save("compare_crops.png")
    print("saved compare_crops.png")


if __name__ == "__main__":
    main()
