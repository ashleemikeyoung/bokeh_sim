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

Re-render a finished photo (shot at f/2 or f/2.8) as it would look on a
full-frame Summicron f/2, Summilux f/1.4 and Noctilux f/0.95, using
Depth Anything V2 for depth. No manual masking.

Usage:
    python bokeh_sim.py photo.jpg                       # APS-C, shot at f/2
    python bokeh_sim.py photo.jpg --sensor m43 --source-f 2.8
    python bokeh_sim.py photo.jpg --sensor ff --lenses summilux
    python bokeh_sim.py photo.jpg --focus 0.50,0.24     # pin focus (fractions of width,height)

Sensor equivalence:
    Depth of field depends on sensor size, so the photo's existing blur is
    treated as that of a full-frame lens at (source f-number x crop factor):
    APS-C 1.5x, M43 2x, FF 1x. Only the extra blur needed to reach each
    target aperture is added (blurs combine in quadrature). A look the photo
    already meets or exceeds is skipped, since blur cannot be removed.
    Assumes the full-frame version is framed identically (e.g. 23mm APS-C
    versus 35mm FF at the same distance).

How the blur works:
    Layered disc blur by depth, premultiplied so the sharp subject does not
    bleed into the background, composited back to front in linear light with
    optional highlight bloom for bokeh discs.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LENSES = {"summicron": 2.0, "summilux": 1.4, "noctilux": 0.95}
SENSOR_CROP = {"ff": 1.0, "apsc": 1.5, "m43": 2.0}


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


def disc_blur(stack, r, max_direct=9.0):
    """Disc blur; big radii are done at reduced resolution for speed."""
    if r <= max_direct:
        return cv2.filter2D(stack, -1, disc_kernel(r), borderType=cv2.BORDER_REFLECT)
    s = r / 8.0
    h, w = stack.shape[:2]
    small = cv2.resize(stack, (max(1, round(w / s)), max(1, round(h / s))), interpolation=cv2.INTER_AREA)
    small = cv2.filter2D(small, -1, disc_kernel(8.0), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


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
            stack = disc_blur(stack, r)
        C, A = stack[..., :3], stack[..., 3:4]
        if acc_c is None:
            acc_c, acc_a = C, A
        else:
            acc_c = C + (1 - A) * acc_c
            acc_a = A + (1 - A) * acc_a

    return acc_c / np.maximum(acc_a, 1e-6)


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------
def read_exif_lens(path):
    """Return (f_number, focal_mm, focal_35mm_equiv); any may be None."""
    try:
        ex = Image.open(path).getexif()
        ifd = ex.get_ifd(0x8769)
        get = lambda k: float(ifd[k]) if ifd.get(k) else None
        return get(0x829D), get(0x920A), get(0xA405)
    except Exception:
        return None, None, None


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_image(path):
    """Return (srgb float32 0-1, exif, icc). JPEG/PNG only."""
    if path.suffix.lower() in (".tif", ".tiff"):
        raise SystemExit("TIFF is not supported. Export a JPEG (sRGB) from your editor and use that.")
    pil = Image.open(path)
    exif, icc = pil.info.get("exif"), pil.info.get("icc_profile")
    return np.array(pil.convert("RGB")).astype(np.float32) / 255.0, exif, icc


def save_image(path, srgb, exif, icc):
    arr = np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)
    kw = {"quality": 95, "subsampling": 0}
    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    Image.fromarray(arr).save(path, **kw)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Re-render a photo as Summicron / Summilux / Noctilux on full frame.")
    ap.add_argument("input")
    ap.add_argument("--source-f", type=float, default=None,
                    help="aperture the photo was shot at (default: read from EXIF)")
    ap.add_argument("--focal-length", type=float, default=None,
                    help="actual focal length in mm (default: read from EXIF)")
    ap.add_argument("--sensor", choices=list(SENSOR_CROP), default=None,
                    help="sensor type (default: worked out from EXIF)")
    ap.add_argument("--crop", type=float, default=None, help="custom crop factor, overrides --sensor")
    ap.add_argument("--lenses", nargs="+", choices=list(LENSES), default=list(LENSES))
    ap.add_argument("--focus", default=None, help="focus point as fractions, e.g. 0.50,0.24")
    ap.add_argument("--out-dir", default=None, help="default: next to the input file")
    ap.add_argument("--strength", type=float, default=0.04,
                    help="blur radius at f/1 for the full depth range, as a fraction of the long side")
    ap.add_argument("--bins", type=int, default=32, help="depth layers (lower = faster)")
    ap.add_argument("--bloom", type=float, default=2.0, help="highlight bokeh boost, 0 to disable")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf",
                    help="use ...-Small-hf for a faster model")
    ap.add_argument("--save-depth", action="store_true", help="also save the depth map PNG")
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    srgb, exif, icc = load_image(src)
    h, w = srgb.shape[:2]
    long_side = max(h, w)
    rgb8 = (srgb * 255 + 0.5).astype(np.uint8)

    ex_f, ex_focal, ex_f35 = read_exif_lens(src)
    f_number = args.source_f or ex_f
    focal = args.focal_length or ex_focal
    if args.crop:
        crop = args.crop
    elif args.sensor:
        crop = SENSOR_CROP[args.sensor]
    elif ex_f35 and ex_focal:
        crop = ex_f35 / ex_focal
    else:
        crop = None

    missing = []
    if not f_number:
        missing.append("aperture (--source-f 2)")
    if not crop:
        missing.append("sensor type (--sensor apsc | m43 | ff)")
    if missing:
        raise SystemExit("No EXIF found, please provide: " + " and ".join(missing))

    n_eff = f_number * crop
    line = f"Source: f/{f_number:g}"
    if focal:
        line += f", {focal:g}mm (about {focal * crop:.0f}mm full-frame equivalent)"
    line += f", {crop:.2f}x crop -> full-frame equivalent f/{n_eff:.2f}"
    print(line)

    print("Estimating depth...")
    disp = estimate_disparity(Image.fromarray(rgb8), args.model)
    if args.save_depth:
        Image.fromarray((disp * 255).astype(np.uint8)).save(out_dir / f"{src.stem}_depth.png")

    if args.focus:
        fx, fy = (float(v) for v in args.focus.split(","))
        d_focus = focus_from_point(disp, fx, fy)
        print(f"Focus plane from point ({fx}, {fy}): {d_focus:.3f}")
    else:
        d_focus = auto_focus(rgb8, disp)
        print(f"Focus plane (auto): {d_focus:.3f}")

    lin = srgb_to_lin(srgb).astype(np.float32)
    k_px = args.strength * long_side
    max_r = 0.06 * long_side

    for name in args.lenses:
        n = LENSES[name]
        if n >= n_eff:
            print(f"Skipping {name} f/{n:g}: the photo is already at or beyond that look.")
            continue
        print(f"Rendering {name} f/{n:g} ...")
        out_lin = render(lin, disp, d_focus, n, n_eff, k_px, args.bins, args.bloom, max_r)
        path = out_dir / f"{src.stem}_{name}-f{n:g}.jpg"
        save_image(path, lin_to_srgb(out_lin), exif, icc)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
