#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
#
# bokeh_sim.py
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
bokeh_sim.py

Simulate wider apertures (f/0.95, 1.0, 1.2, 1.4, 1.8, 2.0) on a finished JPEG,
using Depth Anything V2 for depth. No manual masking.

Setup (Apple Silicon Mac):
    python3 -m venv da2 && source da2/bin/activate
    pip install torch torchvision transformers pillow opencv-python numpy

Usage:
    python bokeh_sim.py photo.jpg
    python bokeh_sim.py photo.jpg --source-f 2.0 --focus 0.50,0.24
    python bokeh_sim.py photo.jpg --apertures 0.95 1.4 --bloom 3 --save-depth

How it works:
  1. Depth Anything V2 gives relative disparity (bigger = closer).
  2. The focus plane is found automatically (sharpest region of the photo),
     or you can pin it with --focus x,y as fractions of width/height.
  3. Real blur size scales with 1/f-number and with the disparity difference
     from the focus plane. The source photo already carries blur from its own
     aperture, so only the *extra* blur is added (blurs combine in quadrature).
     Simulating the source aperture itself returns the original.
  4. The image is split into depth layers, each layer is disc-blurred on its
     own (premultiplied, so sharp subject colour does not bleed into the
     background), then composited back to front. Blur is done in linear light
     with optional highlight bloom so bright points become proper bokeh discs.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_APERTURES = [0.95, 1.0, 1.2, 1.4, 1.8, 2.0]


# --------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------
def estimate_disparity(img, model_id):
    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pipe = pipeline("depth-estimation", model=model_id, device=device)
    out = pipe(img)
    d = out["predicted_depth"].squeeze().float().cpu().numpy()
    w, h = img.size
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, [1, 99])
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Focus plane
# --------------------------------------------------------------------------
def auto_focus(rgb8, disp):
    """Find the depth of the sharpest, most central detail in the photo."""
    g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)  # suppress sensor noise
    s = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    h, w = s.shape
    s = cv2.GaussianBlur(s, (0, 0), max(h, w) / 150)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    center = np.exp(-(((xx - w / 2) / (0.6 * w)) ** 2 + ((yy - h / 2) / (0.6 * h)) ** 2))
    score = s * center
    thr = np.percentile(score, 97)
    return float(np.median(disp[score >= thr]))


def focus_from_point(disp, fx, fy):
    h, w = disp.shape
    cx, cy = int(fx * w), int(fy * h)
    r = max(4, int(0.01 * max(h, w)))
    patch = disp[max(0, cy - r): cy + r, max(0, cx - r): cx + r]
    return float(np.median(patch))


# --------------------------------------------------------------------------
# Colour + blur helpers
# --------------------------------------------------------------------------
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def disc_kernel(r):
    n = int(np.ceil(r))
    y, x = np.mgrid[-n: n + 1, -n: n + 1]
    k = np.clip(r + 0.5 - np.hypot(x, y), 0, 1).astype(np.float32)
    return k / k.sum()


# --------------------------------------------------------------------------
# Render one aperture
# --------------------------------------------------------------------------
def render(lin, disp, d_focus, n_target, n_source, k_px, bins, bloom, max_r):
    gain = np.sqrt(max(0.0, 1 / n_target**2 - 1 / n_source**2))
    if gain == 0:
        return lin

    scale = k_px * gain  # blur radius in px for a disparity gap of 1.0

    # Highlight bloom: boost bright pixels that are going to be blurred.
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hl = np.clip((lum - 0.6) / 0.4, 0, 1)
    hl = hl * hl * (3 - 2 * hl)
    ramp = np.clip(np.abs(disp - d_focus) * scale / 4.0, 0, 1)
    src = lin * (1 + bloom * hl * ramp)[..., None]

    step = 1.0 / bins
    centers = (np.arange(bins) + 0.5) * step
    acc_c = None
    acc_a = None

    for c in centers:  # far (low disparity) to near
        m = np.clip(1 - np.abs(disp - c) / step, 0, 1).astype(np.float32)
        if m.max() < 1e-3:
            continue
        r = min(abs(c - d_focus) * scale, max_r)
        stack = np.dstack([src * m[..., None], m])
        if r >= 0.5:
            stack = cv2.filter2D(stack, -1, disc_kernel(r), borderType=cv2.BORDER_REFLECT)
        C, A = stack[..., :3], stack[..., 3:4]
        if acc_c is None:
            acc_c, acc_a = C, A
        else:
            acc_c = C + (1 - A) * acc_c
            acc_a = A + (1 - A) * acc_a

    return acc_c / np.maximum(acc_a, 1e-6)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Simulate wider apertures with Depth Anything V2.")
    ap.add_argument("input")
    ap.add_argument("--out-dir", default=None, help="default: next to the input file")
    ap.add_argument("--apertures", type=float, nargs="+", default=DEFAULT_APERTURES)
    ap.add_argument("--source-f", type=float, default=2.0, help="aperture the photo was shot at")
    ap.add_argument("--focus", default=None, help="focus point as fractions, e.g. 0.50,0.24")
    ap.add_argument("--strength", type=float, default=0.04,
                    help="blur radius at f/1 for the full depth range, as a fraction of the long side")
    ap.add_argument("--bins", type=int, default=32, help="depth layers (lower = faster)")
    ap.add_argument("--bloom", type=float, default=2.0, help="highlight bokeh boost, 0 to disable")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf",
                    help="use ...-Small-hf for a faster, Apache-2.0 model")
    ap.add_argument("--save-depth", action="store_true", help="also save the depth map PNG")
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(src).convert("RGB")
    exif = Image.open(src).info.get("exif")
    icc = Image.open(src).info.get("icc_profile")
    rgb8 = np.array(img)
    h, w = rgb8.shape[:2]
    long_side = max(h, w)

    print("Estimating depth...")
    disp = estimate_disparity(img, args.model)
    if args.save_depth:
        Image.fromarray((disp * 255).astype(np.uint8)).save(out_dir / f"{src.stem}_depth.png")

    if args.focus:
        fx, fy = (float(v) for v in args.focus.split(","))
        d_focus = focus_from_point(disp, fx, fy)
        print(f"Focus plane from point ({fx}, {fy}): {d_focus:.3f}")
    else:
        d_focus = auto_focus(rgb8, disp)
        print(f"Focus plane (auto): {d_focus:.3f}")

    lin = srgb_to_lin(rgb8.astype(np.float32) / 255.0).astype(np.float32)
    k_px = args.strength * long_side
    max_r = 0.06 * long_side

    for n in args.apertures:
        print(f"Rendering f/{n:g} ...")
        out_lin = render(lin, disp, d_focus, n, args.source_f, k_px, args.bins, args.bloom, max_r)
        out8 = (lin_to_srgb(out_lin) * 255 + 0.5).astype(np.uint8)
        path = out_dir / f"{src.stem}_f{n:g}.jpg"
        kw = {"quality": 95, "subsampling": 0}
        if exif:
            kw["exif"] = exif
        if icc:
            kw["icc_profile"] = icc
        Image.fromarray(out8).save(path, **kw)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
