#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
#
# bokeh_sim.py: re-render a photo with Summicron, Summilux or Noctilux depth of field,
# using Depth Anything V2 depth maps.
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

Re-render a finished photo as it would look with a Summicron f/2, Summilux f/1.4
or Noctilux f/0.95, using Depth Anything V2 for depth. No manual masking.

Usage:
    python bokeh_sim.py photo.jpg                          # same system as the photo
    python bokeh_sim.py photo.jpg --lenses noctilux
    python bokeh_sim.py photo.jpg --target-sensor ff       # restyle as if shot on full frame
    python bokeh_sim.py photo.jpg --sensor m43 --source-f 5 --focal-length 75   # no EXIF
    python bokeh_sim.py photo.jpg --focus 0.50,0.24        # pin focus (fractions of width,height)

Same system vs another sensor:
    By default the look stays on the photo's own sensor: an M43 f/5 photo becomes
    M43 f/2, f/1.4 or f/0.95. --target-sensor ff|apsc|m43|mf (or --target-crop) instead
    asks "what if this had been shot on that sensor at that aperture?", which compares
    them on a full-frame-equivalent basis (f-number x crop factor). A look the photo
    already meets or exceeds is skipped, since blur cannot be removed.
    Assumes the same framing (e.g. 23mm APS-C versus 35mm FF at the same distance).

How the blur works:
    Layered disc blur by depth, premultiplied so the sharp subject does not
    bleed into the background, composited back to front in linear light with
    optional highlight bloom for bokeh discs. Only the extra blur needed is added
    (blurs combine in quadrature).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LENSES = {"summicron": 2.0, "summilux": 1.4, "noctilux": 0.95}
SENSOR_CROP = {"ff": 1.0, "apsc": 1.5, "m43": 2.0, "mf": 0.79}  # mf = 44x33 medium format (GFX, X2D)
FF_LONG_MM = 36.0  # long side of a full-frame sensor
METRIC_MODELS = {
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


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


def estimate_metric_depth(img, model_id):
    """Depth in metres (larger = farther) from a metric Depth Anything V2 model."""
    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Depth model device: {device.type}")
    pipe = pipeline("depth-estimation", model=model_id, device=device)
    z = pipe(img)["predicted_depth"].squeeze().float().cpu().numpy()
    w, h = img.size
    z = cv2.resize(z, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(z, 0.2, 300.0).astype(np.float32)


DEPTH_MODELS = ["da2", "da2-metric", "depthpro", "moge2", "moge3", "moge3-mlx"]
MOGE_REPOS = {"moge2": "Ruicheng/moge-2-vitl", "moge3": "Ruicheng/moge-3-vitl"}
MOGE3_MLX_REPO = "mlx-community/moge-3-vitl-mlx-fp32"
MOGE_INSTALL = ("python -m pip install --no-deps git+https://github.com/microsoft/MoGe.git "
                "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37")


def _run_on_best_device(fn):
    """Run fn(device) on mps/cuda when available; if that fails, say so and retry on the CPU."""
    import torch

    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"Depth model device: {dev.type}")
    try:
        return fn(dev)
    except Exception as e:
        if dev.type == "cpu":
            raise
        print(f"  {dev.type} failed ({type(e).__name__}: {str(e)[:100]}); retrying on the CPU, which is slower")
        return fn(torch.device("cpu"))


def depth_depthpro(img, f_true_px=None):
    """Apple Depth Pro via transformers: sharp metric depth in metres."""
    import torch
    from transformers import AutoImageProcessor, DepthProForDepthEstimation

    def run(dev):
        proc = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
        model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(dev).eval()
        inputs = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs)
        return proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]

    post = _run_on_best_device(run)
    z = post["predicted_depth"].squeeze().float().cpu().numpy()
    f_est = post.get("focal_length")
    if f_true_px and f_est is not None:
        f_est = float(torch.as_tensor(f_est).reshape(-1)[0])
        ratio = f_true_px / f_est
        if 0.4 < ratio < 2.5:
            z = z * ratio   # Depth Pro depth scales with the focal length it assumed; use the real one
            print(f"  Depth Pro assumed a {f_est:.0f} px focal length; your lens is {f_true_px:.0f} px, depths scaled by {ratio:.2f}")
        else:
            print(f"  Depth Pro focal length ({f_est:.0f} px) and yours ({f_true_px:.0f} px) disagree a lot; left unscaled")
    return z


def depth_moge(img, name, fov_x_deg=None, weights=None):
    """MoGe-2 / MoGe-3 (microsoft/MoGe): metric depth in metres. Uses your real field of view when known."""
    try:
        if name == "moge3":
            from moge.model.v3 import MoGeModel
        else:
            from moge.model.v2 import MoGeModel
    except ImportError as e:
        extra = ("\nMoGe-3 needs flex-gemm, which is built for NVIDIA GPUs and is unlikely to work on a Mac. "
                 "On a Mac use --depth-model moge3-mlx (or moge2)." if name == "moge3" else "")
        raise SystemExit(f"MoGe could not be imported ({e}). Install it with:\n  {MOGE_INSTALL}{extra}")
    import torch
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    source = MOGE_REPOS[name]
    if weights:
        if not os.path.isfile(os.path.expanduser(weights)):
            raise SystemExit(f"--depth-weights file not found: {weights}")
        source = os.path.expanduser(weights)
        print(f"  loading MoGe weights from {source} (no Hugging Face access needed)")

    def run(dev):
        model = MoGeModel.from_pretrained(source).to(dev).eval()
        t = torch.tensor(arr, dtype=torch.float32, device=dev).permute(2, 0, 1)
        kw = {"use_fp16": dev.type == "cuda"}
        if fov_x_deg:
            kw["fov_x"] = float(fov_x_deg)
        with torch.no_grad():
            return model.infer(t, **kw)

    out = _run_on_best_device(run)
    return out["depth"].float().cpu().numpy()


def depth_moge3_mlx(img, fov_x_deg=None):
    """MoGe-3 on Apple silicon through the MLX port (mlx-vlm >= 0.7.8): metric depth in metres."""
    try:
        from mlx_vlm import load
        from mlx_vlm.models.moge3.generate import MoGe3Predictor
    except ImportError as e:
        raise SystemExit(f"The MLX MoGe-3 port could not be imported ({e}). Install or upgrade it with:\n"
                         "  python -m pip install -U mlx-vlm      (version 0.7.8 or newer)")
    print("Depth model device: Apple silicon (MLX)")
    model, processor = load(MOGE3_MLX_REPO)
    predictor = MoGe3Predictor(model, processor)
    kw = {"fov_x": float(fov_x_deg)} if fov_x_deg else {}
    out = predictor.infer(np.asarray(img.convert("RGB")), **kw)
    return np.asarray(out["depth"], dtype=np.float32)


def metric_depth(name, img, metric_model, fov_x_deg=None, f_true_px=None, weights=None):
    """Depth in metres (larger = farther) from the chosen model."""
    if name == "depthpro":
        z = depth_depthpro(img, f_true_px)
    elif name == "moge3-mlx":
        z = depth_moge3_mlx(img, fov_x_deg)
    elif name in MOGE_REPOS:
        z = depth_moge(img, name, fov_x_deg, weights)
    else:
        z = estimate_metric_depth(img, METRIC_MODELS[metric_model])
    z = np.nan_to_num(z, nan=300.0, posinf=300.0, neginf=0.2)   # masked pixels (sky etc.) count as far away
    return np.clip(z, 0.2, 300.0).astype(np.float32)


def coc_scale(f_mm, n, s_m, long_px, sensor_long_mm):
    """
    Thin-lens blur: radius in pixels per 1/metre of disparity gap from the focus plane.
    Circle of confusion diameter c = f^2 * s / (N * (s - f)) * |1/s - 1/z|   (all in metres).
    """
    f = f_mm / 1000.0
    s = max(s_m, f * 1.05)
    c_m = f * f * s / (n * (s - f))
    return 0.5 * c_m * 1000.0 * (long_px / sensor_long_mm)


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
def render(lin, disp, d_focus, n_target, n_source, k_px, bins, bloom, max_r, scale_px=None):
    if scale_px is None:
        gain = np.sqrt(max(0.0, 1 / n_target**2 - 1 / n_source**2))
        if gain == 0:
            return lin
        scale = k_px * gain  # blur radius in px for a disparity gap of 1.0
    else:
        scale = scale_px  # physical mode: already in px per unit of normalised disparity

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


def guess_sensor_from_camera(path):
    """Best-effort sensor size from the camera make and model. Returns (key or None, make, model)."""
    try:
        ex = Image.open(path).getexif()
        make = str(ex.get(0x010F) or "").strip()
        model = str(ex.get(0x0110) or "").strip()
    except Exception:
        return None, "", ""
    import re
    m = (make + " " + model).upper()
    if "FUJIFILM" in m:
        if re.search(r"\b(X-HALF|X10|X20|X30|XQ1|XQ2|X-S1)\b", model.upper()) or "FINEPIX" in m:
            return None, make, model          # compacts with smaller sensors: do not guess
        return ("mf" if "GFX" in m else "apsc"), make, model
    return None, make, model


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
    ap = argparse.ArgumentParser(description="Re-render a photo as Summicron / Summilux / Noctilux, in its own system or for another sensor size.")
    ap.add_argument("input")
    ap.add_argument("--source-f", type=float, default=None,
                    help="aperture the photo was shot at (default: read from EXIF)")
    ap.add_argument("--focal-length", type=float, default=None,
                    help="actual focal length in mm (default: read from EXIF)")
    ap.add_argument("--sensor", choices=list(SENSOR_CROP), default=None,
                    help="sensor type (default: worked out from EXIF)")
    ap.add_argument("--crop", type=float, default=None, help="custom crop factor, overrides --sensor")
    ap.add_argument("--target-sensor", choices=["same"] + list(SENSOR_CROP), default="same",
                    help="sensor the look is rendered for (default: same system as the photo)")
    ap.add_argument("--target-crop", type=float, default=None,
                    help="custom target crop factor, overrides --target-sensor")
    ap.add_argument("--assume-sensor", choices=list(SENSOR_CROP), default=None,
                    help="pretend the photo was shot on this sensor (same framing, same f-number); the "
                         "default look target then also stays on it")
    ap.add_argument("--blur-scale", type=float, default=1.0,
                    help="multiplier on the added blur: 1 = true to the optics, below 1 subtler, above 1 exaggerated")
    ap.add_argument("--physical", action="store_true",
                    help="true-to-optics blur from metric depth, focal length, f-number and sensor size "
                         "(ignores --strength)")
    ap.add_argument("--depth-model", choices=DEPTH_MODELS, default=None,
                    help="da2 = Depth Anything V2 relative depth (default); with --physical the default is "
                         "da2-metric. depthpro = Apple Depth Pro, moge2 / moge3 = MoGe (extra install; "
                         "moge3 needs an NVIDIA GPU), moge3-mlx = MoGe-3 on Apple silicon via mlx-vlm. "
                         "All but da2 give metric depth.")
    ap.add_argument("--depth-weights", default=None,
                    help="path to a local MoGe model.pt, so moge2 runs with no Hugging Face access")
    ap.add_argument("--metric-model", choices=list(METRIC_MODELS), default="outdoor",
                    help="outdoor or indoor variant when the depth model is da2-metric (default: outdoor)")
    ap.add_argument("--focus-distance", type=float, default=None,
                    help="with --physical: real distance to the focus point in metres (fixes the depth scale)")
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
    if not crop:
        key, mk, md = guess_sensor_from_camera(src)
        if key:
            crop = SENSOR_CROP[key]
            print(f"The EXIF has no 35mm-equivalent focal length, so assuming the {mk} {md} is {key} ({crop:g}x crop).")

    missing = []
    if not f_number:
        missing.append("aperture (--source-f 2)")
    if not crop:
        missing.append("sensor type (--sensor apsc | m43 | ff)")
    if missing:
        raise SystemExit("The photo's EXIF does not include: " + " and ".join(missing) + ". Give them as flags.")

    crop_src = SENSOR_CROP[args.assume_sensor] if args.assume_sensor else crop   # system the blur model assumes
    n_eff = f_number * crop_src
    if args.target_crop:
        crop_t, tgt_label = args.target_crop, f"{args.target_crop:g}x crop"
    elif args.target_sensor != "same":
        crop_t, tgt_label = SENSOR_CROP[args.target_sensor], args.target_sensor
    else:
        crop_t, tgt_label = crop_src, "same system"
    line = f"Source: f/{f_number:g}"
    if focal:
        line += f", {focal:g}mm (about {focal * crop:.0f}mm full-frame equivalent)"
    line += f", {crop:.2f}x crop -> full-frame equivalent f/{n_eff:.2f}"
    print(line)
    if args.assume_sensor:
        print(f"Pretending the photo was shot on {args.assume_sensor} at f/{f_number:g}, same framing")

    dm = args.depth_model or ("da2-metric" if args.physical else "da2")
    if args.physical and dm == "da2":
        dm = "da2-metric"
        print("--physical needs metric depth, so using da2-metric.")
    use_metric = dm != "da2"
    fov_x = f_true_px = None
    if focal:   # real lens and sensor -> real field of view and focal length in pixels
        width_mm = (FF_LONG_MM / crop) * w / long_side
        fov_x = float(np.degrees(2 * np.arctan(width_mm / (2 * focal))))
        f_true_px = focal * w / width_mm
    print(f"Estimating depth with {dm}...")
    dmax = None
    if use_metric:
        if args.physical and not focal:
            raise SystemExit("--physical needs the real focal length: add --focal-length 35 (mm).")
        if not args.physical:
            print("  (metric model used for the depth map only; blur strength still comes from --strength)")
        if args.depth_weights and dm not in MOGE_REPOS:
            print(f"  (--depth-weights only applies to moge2 / moge3, ignored for {dm})")
        z = metric_depth(dm, Image.fromarray(rgb8), args.metric_model, fov_x, f_true_px, args.depth_weights)
        inv = 1.0 / z                                     # true disparity, 1/m
        dmax = float(np.percentile(inv, 99.5))
        disp = np.clip(inv / dmax, 0, 1).astype(np.float32)   # normalised 0-1, near = 1
    else:
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
    k_px = args.strength * long_side * args.blur_scale
    max_r = 0.06 * long_side
    s_m = None
    if args.physical:
        s_m = float(np.clip(1.0 / max(d_focus * dmax, 1e-6), 0.3, 300.0))
        print(f"Physical mode: the model puts the focus point at about {s_m:.1f} m")
        if args.focus_distance:
            dmax *= s_m / args.focus_distance      # rescale all distances so the focus point is where you say
            s_m = args.focus_distance
            print(f"  using your focus distance of {s_m:g} m instead")
        max_r = 0.15 * long_side
        sensor_long_s = FF_LONG_MM / crop_src

    for name in args.lenses:
        n = LENSES[name]
        n_t_eff = n * crop_t  # full-frame-equivalent f-number of the target look
        if n_t_eff >= n_eff:
            print(f"Skipping {name} f/{n:g} ({tgt_label}): the photo is already at or beyond that look.")
            continue
        print(f"Rendering {name} f/{n:g} on {tgt_label} (full-frame equivalent f/{n_t_eff:.2f}) ...")
        scale_px = None
        if args.physical:
            f_t = focal * crop / crop_t                       # same framing on the target sensor
            k_s = coc_scale(focal * crop / crop_src, f_number, s_m, long_side, sensor_long_s)
            k_t = coc_scale(f_t, n, s_m, long_side, FF_LONG_MM / crop_t)
            if k_t <= k_s:
                print(f"  skipped: the photo already has at least that much blur ({k_s / s_m:.1f} px at infinity)")
                continue
            add = float(np.sqrt(k_t**2 - k_s**2)) * args.blur_scale
            scale_px = add * dmax
            final = float(np.hypot(k_s, add)) / s_m
            note = f", blur x{args.blur_scale:g}" if args.blur_scale != 1 else ""
            print(f"  background blur radius at infinity: photo{' (assumed)' if args.assume_sensor else ''} {k_s / s_m:.1f} px -> {name} {final:.1f} px "
                  f"(target lens: {f_t:.0f}mm f/{n:g}{note})")
        out_lin = render(lin, disp, d_focus, n_t_eff, n_eff, k_px, args.bins, args.bloom, max_r, scale_px)
        suffix = "" if (args.target_sensor == "same" and not args.target_crop) else f"_{args.target_sensor if not args.target_crop else 'crop' + str(args.target_crop)}"
        path = out_dir / f"{src.stem}_{name}-f{n:g}{suffix}.jpg"
        save_image(path, lin_to_srgb(out_lin), exif, icc)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
