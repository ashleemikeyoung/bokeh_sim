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

__version__ = "0.1.0"
LENSES = {"summicron": 2.0, "summilux": 1.4, "noctilux": 0.95}
SENSOR_CROP = {"ff": 1.0, "apsc": 1.5, "m43": 2.0}
GRAIN_SIGMA = 0.3  # fineness of synthetic grain, matched to Fuji X-series JPEG grain


# --------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------
def estimate_disparity(img, model_id, force_cpu=False):
    import torch
    from transformers import pipeline

    if force_cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Depth model device: {device.type}")
    pipe = pipeline("depth-estimation", model=model_id, device=device)
    out = pipe(img)
    d = out["predicted_depth"].squeeze().float().cpu().numpy()
    w, h = img.size
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, [1, 99])
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Depth refinement + eye focus
# --------------------------------------------------------------------------
def refine_depth(rgb8, disp):
    """Snap the (coarse) depth edges to real image edges with a guided filter."""
    h, w = disp.shape
    r = max(4, int(0.006 * max(h, w)))
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    guide = cv2.GaussianBlur(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0, (0, 0), 1.0)
    mi, mp = box(guide), box(disp)
    cov = box(guide * disp) - mi * mp
    var = box(guide * guide) - mi * mi
    a = cov / (var + 1e-3)
    b = mp - a * mi
    return np.clip(box(a) * guide + box(b), 0, 1).astype(np.float32)


class EyeDetectionUnavailable(Exception):
    pass


def detect_eye_focus(rgb8, disp):
    """Find a cat or human face, take depth at the nearer eye. Returns (disparity, label) or None."""
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(getattr(cv2, "data", None), "haarcascades"):
        raise EyeDetectionUnavailable
    g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    min_sz = int(0.08 * max(h, w))
    d = cv2.data.haarcascades
    for xml, label, eyes in (
        ("haarcascade_frontalcatface_extended.xml", "cat", ((0.28, 0.40), (0.72, 0.40))),
        ("haarcascade_frontalface_default.xml", "face", ((0.30, 0.38), (0.70, 0.38))),
    ):
        det = cv2.CascadeClassifier(d + xml)
        if det.empty():
            continue
        faces = det.detectMultiScale(g, 1.05, 6, minSize=(min_sz, min_sz))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            r = max(3, int(0.03 * fw))
            vals = []
            for ex, ey in eyes:
                cx, cy = int(x + ex * fw), int(y + ey * fh)
                patch = disp[max(0, cy - r): cy + r, max(0, cx - r): cx + r]
                vals.append(float(np.median(patch)))
            return max(vals), label  # nearer eye wins
    return None


def pick_focus(rgb8):
    """Show the photo; click the eye (or anything to keep sharp), then press any key."""
    h, w = rgb8.shape[:2]
    sc = min(1.0, 1400 / max(h, w))
    view = cv2.resize(np.ascontiguousarray(rgb8[..., ::-1]), (int(w * sc), int(h * sc)))
    pt = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt["xy"] = (x / view.shape[1], y / view.shape[0])

    title = "Click the eye, then press any key"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    while True:
        shown = view.copy()
        if "xy" in pt:
            cx, cy = int(pt["xy"][0] * view.shape[1]), int(pt["xy"][1] * view.shape[0])
            cv2.circle(shown, (cx, cy), 12, (0, 255, 0), 2)
        cv2.imshow(title, shown)
        if cv2.waitKey(30) != -1 and "xy" in pt:
            break
    cv2.destroyAllWindows()
    return pt["xy"]


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
def render(lin, disp, d_focus, n_target, n_source, k_px, bins, bloom, max_r, zone=0.0):
    gain = np.sqrt(max(0.0, 1 / n_target**2 - 1 / n_source**2))
    if gain == 0:
        return lin, np.zeros(lin.shape[:2], np.float32)

    scale = k_px * gain  # blur radius in px for a disparity gap of 1.0

    # In-focus zone: everything within +/- zone of the focus plane stays perfectly
    # sharp (this is what keeps the whole face, not just one slice of it, crisp).
    delta = disp - d_focus
    disp = d_focus + np.sign(delta) * np.maximum(np.abs(delta) - zone, 0)

    # Highlight bloom: only true bright points (specular highlights), not white fur.
    lum = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hl = np.clip((lum - 0.85) / 0.15, 0, 1)
    hl = hl * hl * (3 - 2 * hl)
    r_map = np.minimum(np.abs(disp - d_focus) * scale, max_r).astype(np.float32)
    ramp = np.clip(r_map / 4.0, 0, 1)
    src = lin * (1 + bloom * hl * ramp)[..., None]

    # Depth layers whose centres line up exactly with the focus plane, so
    # in-focus pixels land on a layer with zero blur and are left untouched.
    step = 1.0 / bins
    k_lo = int(np.floor((0 - d_focus) / step))
    k_hi = int(np.ceil((1 - d_focus) / step))
    centers = d_focus + step * np.arange(k_lo, k_hi + 1)

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

    return acc_c / np.maximum(acc_a, 1e-6), r_map


# --------------------------------------------------------------------------
# Grain
# --------------------------------------------------------------------------
def _noise_level(g):
    """Rough sensor-noise estimate: quietest quarter of 64px tiles, robust MAD of the high-pass."""
    res = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    t = 64
    h, w = res.shape
    h2, w2 = h // t * t, w // t * t
    tiles = np.abs(res[:h2, :w2]).reshape(h2 // t, t, w2 // t, t).transpose(0, 2, 1, 3).reshape(-1, t * t)
    return float(np.percentile(1.4826 * np.median(tiles, axis=1), 25))


def add_grain(out_srgb, orig_srgb, r_map, strength=1.0):
    """Blur averages sensor grain away; put matching grain back where blur was applied."""
    gray = cv2.cvtColor(orig_srgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    rng = np.random.default_rng(0)
    unit = cv2.GaussianBlur(rng.standard_normal(gray.shape).astype(np.float32), (0, 0), GRAIN_SIGMA)
    unit /= unit.std()
    amp = _noise_level(gray) / max(_noise_level(unit), 1e-8)
    w = np.clip(r_map / 2.0, 0, 1)
    return out_srgb + (strength * amp * w * unit)[..., None]


# --------------------------------------------------------------------------
# Fine-detail protection (whiskers, single hairs)
# --------------------------------------------------------------------------
def _line_kernel(theta, off, sx, sy, n):
    r = np.arange(-n, n + 1, dtype=np.float32)
    X, Y = np.meshgrid(r, r)
    u = X * np.cos(theta) + Y * np.sin(theta)            # along the line
    v = -X * np.sin(theta) + Y * np.cos(theta) - off     # across the line
    k = np.exp(-(u**2) / (2 * sx**2) - (v**2) / (2 * sy**2))
    return (k / k.sum()).astype(np.float32)


def line_response(L, angles=12, sx=5.0, sy=0.9, side=2.8):
    """Thin bright OR dark lines. The line centre must beat BOTH sides, so plain edges score ~0."""
    n = int(np.ceil(3 * sx))
    best = np.zeros_like(L)
    for i in range(angles):
        th = np.pi * i / angles
        c = cv2.filter2D(L, -1, _line_kernel(th, 0, sx, sy, n), borderType=cv2.BORDER_REFLECT)
        a = cv2.filter2D(L, -1, _line_kernel(th, +side, sx, sy, n), borderType=cv2.BORDER_REFLECT)
        b = cv2.filter2D(L, -1, _line_kernel(th, -side, sx, sy, n), borderType=cv2.BORDER_REFLECT)
        best = np.maximum(best, np.maximum(np.minimum(c - a, c - b), np.minimum(a - c, b - c)))
    return np.maximum(best, 0)


def fine_detail_mask(srgb, disp, d_focus, zone, reach=0.14):
    """
    Depth maps cannot see whiskers (1-3 px wide), so they get blurred away as background.
    Find crisp thin lines in the original and return how strongly to restore them (0..1),
    fading out with distance from the in-focus subject.
    """
    h, w = disp.shape
    long_side = max(h, w)
    L = cv2.cvtColor(srgb.astype(np.float32), cv2.COLOR_RGB2GRAY)

    # threshold calibrated on synthetic grain with the same strength as the photo's
    amp = _noise_level(L)
    rng = np.random.default_rng(1)
    u = cv2.GaussianBlur(rng.standard_normal((640, 640)).astype(np.float32), (0, 0), GRAIN_SIGMA)
    u /= u.std()
    thr = float(np.percentile(line_response(0.5 + amp * u / max(_noise_level(u), 1e-8)), 99.7))

    R = line_response(L)
    a = np.clip((R - thr) / max(thr, 1e-6), 0, 1)
    a = cv2.GaussianBlur(cv2.dilate(a, np.ones((3, 3), np.uint8)), (0, 0), 0.7)

    # only near the in-focus subject
    near = (np.abs(disp - d_focus) <= zone).astype(np.uint8)
    dist = cv2.distanceTransform(1 - near, cv2.DIST_L2, 5)
    gate = np.clip((1 - dist / (reach * long_side)) * 1.6, 0, 1)  # full strength for the first third, then fade
    return (a * gate).astype(np.float32)


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
    ap.add_argument("--version", action="version", version=f"bokeh_sim {__version__}")
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
    ap.add_argument("--pick", action="store_true", help="click the focus point (the eye) on the photo")
    ap.add_argument("--focus-zone", type=float, default=0.08,
                    help="depth band around the eye that stays fully sharp (bigger = more of the face sharp)")
    ap.add_argument("--grain", type=float, default=1.0,
                    help="re-add sensor grain to blurred areas to match the original (0 = off)")
    ap.add_argument("--detail", type=float, default=1.0,
                    help="keep whiskers and other hair-thin detail sharp near the subject (0 = off)")
    ap.add_argument("--no-refine", action="store_true", help="skip depth edge refinement")
    ap.add_argument("--out-dir", default=None, help="default: next to the input file")
    ap.add_argument("--strength", type=float, default=0.04,
                    help="blur radius at f/1 for the full depth range, as a fraction of the long side")
    ap.add_argument("--bins", type=int, default=32, help="depth layers (lower = faster)")
    ap.add_argument("--bloom", type=float, default=1.0, help="highlight bokeh boost, 0 to disable")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf",
                    help="use ...-Small-hf for a faster model")
    ap.add_argument("--cpu", action="store_true", help="run the depth model on the CPU only")
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
    disp = estimate_disparity(Image.fromarray(rgb8), args.model, args.cpu)
    if args.save_depth:
        Image.fromarray((disp * 255).astype(np.uint8)).save(out_dir / f"{src.stem}_depth.png")

    if not args.no_refine:
        disp = refine_depth(rgb8, disp)

    if args.focus:
        fx, fy = (float(v) for v in args.focus.split(","))
        d_focus = focus_from_point(disp, fx, fy)
        print(f"Focus plane from point ({fx}, {fy}): {d_focus:.3f}")
    elif args.pick:
        fx, fy = pick_focus(rgb8)
        d_focus = focus_from_point(disp, fx, fy)
        print(f"Focus plane from click ({fx:.2f}, {fy:.2f}): {d_focus:.3f}")
    else:
        try:
            found = detect_eye_focus(rgb8, disp)
        except EyeDetectionUnavailable:
            found = None
            print(f"Note: this OpenCV build (v{cv2.__version__}) has no face detector, so eye detection is off. "
                  "Use --pick or --focus x,y to aim at the eye.")
        if found:
            d_focus, label = found
            print(f"Eye focus: {label} face found, focusing on the nearer eye ({d_focus:.3f})")
        else:
            d_focus = auto_focus(rgb8, disp)
            print(f"No face found, focus plane (auto): {d_focus:.3f}")

    keep = None
    if args.detail > 0:
        print("Finding whiskers and fine detail...")
        keep = np.clip(fine_detail_mask(srgb, disp, d_focus, args.focus_zone) * args.detail, 0, 1)

    lin = srgb_to_lin(srgb).astype(np.float32)
    k_px = args.strength * long_side
    max_r = 0.06 * long_side

    for name in args.lenses:
        n = LENSES[name]
        if n >= n_eff:
            print(f"Skipping {name} f/{n:g}: the photo is already at or beyond that look.")
            continue
        print(f"Rendering {name} f/{n:g} ...")
        out_lin, r_map = render(lin, disp, d_focus, n, n_eff, k_px, args.bins, args.bloom, max_r, args.focus_zone)
        if keep is not None:
            out_lin = out_lin * (1 - keep[..., None]) + lin * keep[..., None]
            r_map = r_map * (1 - keep)
        out_srgb = lin_to_srgb(out_lin)
        if args.grain > 0:
            out_srgb = add_grain(out_srgb, srgb, r_map, args.grain)
        path = out_dir / f"{src.stem}_{name}-f{n:g}.jpg"
        save_image(path, out_srgb, exif, icc)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
