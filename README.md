# bokeh-sim

Re-render a finished photo as a **Summicron f/2**, **Summilux f/1.4** or **Noctilux f/0.95**. A depth model
estimates how far away each part of the scene is, and the background is blurred by depth. No manual masking.
The subject keeps the colour, contrast and sharpness of your JPEG.

The blur can come from two places:

- **Relative mode** (default): a relative depth map, scaled by `--strength`. Quick, and easy to tune by eye.
- **Physical mode** (`--physical`): metric depth in metres plus your real focal length, f-number, sensor size and
  focus distance, using the thin-lens formula. The blur size is what that lens would really produce.

~~~
python bokeh_sim.py photo.jpg
python bokeh_sim.py photo.jpg --physical --depth-model moge2
python bokeh_sim.py photo.jpg --physical --depth-model moge2 --target-sensor ff --lenses noctilux
~~~

## Install

Python 3.10 or newer. Create an environment and install the core requirements. Using conda (any environment name
works):

~~~
conda create -n bokeh-sim python=3.11
conda activate bokeh-sim
python -m pip install -r requirements.txt
~~~

That is everything you need for the default Depth Anything V2 models and for Apple Depth Pro. The extra files below
are only for the optional MoGe models.

### Which requirements file do I need?

| File | Install it when | Command |
|---|---|---|
| `requirements.txt` | Always. These are the core packages the scripts import | `python -m pip install -r requirements.txt` |
| `requirements-moge.txt` | You want `--depth-model moge2` | `python -m pip install --no-deps -r requirements-moge.txt` |
| `requirements-mlx.txt` | You are on an Apple silicon Mac and want `--depth-model moge3-mlx` | `python -m pip install -r requirements-mlx.txt` |
| `requirements-tested.txt` | Not needed to run. It lists the exact versions the project was last run with, for reference | `python -m pip install -r requirements-tested.txt` only if you want to reproduce that setup exactly |

A few notes:

- **MoGe uses `--no-deps` on purpose.** MoGe's own requirements pull in heavy extras (a web UI toolkit, a second copy
  of OpenCV, and an NVIDIA-only library) that this project does not use. The flag installs just MoGe and the one helper
  package it needs.
- **`requirements-mlx.txt` can upgrade `transformers`.** MoGe-3 on MLX needs `transformers` 5.14 or newer. The line is
  skipped automatically on anything other than Apple silicon.
- **Minimum versions.** `requirements.txt` lists minimums so it works inside an environment you already have.
  `requirements-tested.txt` has the exact versions.

### Which depth model needs what?

| `--depth-model` | What it is | Install |
|---|---|---|
| `da2` | Depth Anything V2, relative depth (default without `--physical`) | `requirements.txt` |
| `da2-metric` | Depth Anything V2, metric depth | `requirements.txt` |
| `depthpro` | Apple Depth Pro, metric depth | `requirements.txt` |
| `moge2` | MoGe-2, metric depth | `requirements.txt` and `requirements-moge.txt` |
| `moge3-mlx` | MoGe-3 on Apple silicon, metric depth | `requirements.txt` and `requirements-mlx.txt` |
| `moge3` | MoGe-3 with PyTorch | NVIDIA GPU only, not covered by these files |

The model weights are not installed with the packages. Each model downloads from Hugging Face the first time you use
it and is cached after that. Your photos never leave your machine, only the model files are downloaded.

The script uses Apple's GPU (`mps`) or CUDA when available. If the GPU run fails it retries on the CPU, which is slower.

### Running offline

After the first download the weights are cached. Set `HF_HUB_OFFLINE=1` to stop any network access. MoGe-2 can also
load a local file, so it needs no Hugging Face access at all:

~~~
HF_HUB_OFFLINE=1 python bokeh_sim.py photo.jpg --physical --depth-model moge2 --depth-weights ~/models/moge2/model.pt
~~~

## Usage

Export your edited photo as an sRGB JPEG first. TIFF is not supported.

~~~
python bokeh_sim.py photo.jpg
~~~

The script reads the aperture, focal length and crop factor from the EXIF data and prints what it found. Check that
line. Results are written next to the photo, for example `photo_summilux-f1.4.jpg`.

If the EXIF is missing, give the details yourself:

~~~
python bokeh_sim.py photo.jpg --source-f 2 --sensor apsc --focal-length 23
~~~

For Fujifilm cameras whose EXIF lacks the 35mm-equivalent focal length, the sensor is worked out from the camera
model (X-series is APS-C, GFX is medium format). Other brands, and small-sensor Fuji compacts, need `--sensor`.

### Looks and sensors

Looks are compared on a full-frame-equivalent basis, which is the f-number times the sensor's crop factor.
Smaller means shallower.

| Look | M43 (2.0x) | APS-C (1.5x) | Full frame (1.0x) | Medium format (0.79x) |
|---|---|---|---|---|
| Summicron f/2 | 4.0 | 3.0 | 2.0 | 1.6 |
| Summilux f/1.4 | 2.8 | 2.1 | 1.4 | 1.1 |
| Noctilux f/0.95 | 1.9 | 1.4 | 0.95 | 0.75 |

By default a look stays on the photo's own sensor. `--target-sensor ff|apsc|m43|mf` renders it as if the photo had
been shot on that sensor at that aperture, with the same framing. A look the photo already meets or exceeds is
skipped, because blur cannot be removed.

### Physical mode

~~~
python bokeh_sim.py photo.jpg --physical --depth-model moge2 --focus 0.54,0.40
~~~

- Needs the real focal length, from EXIF or `--focal-length`.
- Blur depends on the distance to the focus point. If the model misjudges it, give the real distance in metres
  with `--focus-distance`.
- `--blur-scale` multiplies the added blur: 1 is true to the optics, below 1 is subtler, above 1 is exaggerated.
- `--strength` is ignored in this mode.
- `--depth-model da2-metric` is the Depth Anything V2 alternative. Use `--metric-model indoor` for rooms.

### Options

| Option | What it does |
|---|---|
| `--lenses summicron summilux noctilux` | Which looks to render (default: all three) |
| `--source-f 2` | Aperture the photo was shot at (default: EXIF) |
| `--focal-length 23` | Real focal length in mm (default: EXIF) |
| `--sensor ff\|apsc\|m43\|mf` | Sensor the photo was shot on (default: EXIF) |
| `--crop 1.5` | Custom source crop factor |
| `--target-sensor same\|ff\|apsc\|m43\|mf` | Sensor the look is rendered for (default: `same`) |
| `--target-crop 0.62` | Custom target crop factor |
| `--assume-sensor ff` | Pretend the photo was shot on this sensor at the same framing and f-number |
| `--focus 0.54,0.40` | Focus point as fractions of width and height from the top left |
| `--physical` | Blur from real optics and metric depth |
| `--depth-model NAME` | `da2`, `da2-metric`, `depthpro`, `moge2`, `moge3`, `moge3-mlx` |
| `--depth-weights PATH` | Local MoGe model file, for offline use |
| `--metric-model outdoor\|indoor` | Variant used by `da2-metric` |
| `--focus-distance 7` | Real distance to the focus point in metres (physical mode) |
| `--blur-scale 1.5` | Multiplier on the added blur |
| `--strength 0.04` | Blur amount in relative mode |
| `--bloom 0` | Highlight bokeh boost, 0 turns it off. Use 0 for white subjects and bright skies |
| `--bins 32` | Number of depth layers, lower is faster |
| `--save-depth` | Also save the depth map as a PNG |
| `--out-dir DIR` | Where to write results |

### Choosing the focus point

`--focus x,y` takes two fractions of the photo's width and height, measured from the top left. `0.50,0.50` is the
centre. Without it the script picks the sharpest, most central detail, which is not always your subject.

## Comparing depth models

Run the same photo through each model into its own folder, then line the results up:

~~~
python bokeh_sim.py photo.jpg --physical --depth-model moge2 --lenses summilux --focus 0.54,0.40 --save-depth --out-dir cmp-moge2
python bokeh_sim.py photo.jpg --physical --depth-model depthpro --lenses summilux --focus 0.54,0.40 --save-depth --out-dir cmp-depthpro
python compare_depth_models.py photo.jpg moge2=cmp-moge2 depthpro=cmp-depthpro
~~~

This writes `compare_depth.png` (the depth maps) and `compare_crops.png` (zoomed crops of the original and each
render). Add `--crop name:x0,y0,x1,y1`, with fractions of the photo, to inspect other areas. On one indoor
portrait, the MoGe models gave the cleanest outlines and the Depth Anything V2 metric model had the least
convincing structure. Results will vary by photo, so try your own.

## Limitations

- Thin structures such as whiskers and single hairs are not resolved by any depth model and get blurred away.
- The result is an approximation. It does not copy a real lens's rendering character, such as glow, swirl or
  cat's-eye bokeh.
- Metric depth is estimated, so the blur size is only as accurate as the distance the model guesses.
- Blurred areas lose the original sensor grain.

## Versions

Earlier versions are kept as git tags, so you can check any of them out with `git checkout <tag>`.

| Tag | What it is |
|---|---|
| `pre-eye-detection` | The first complete version: depth-based blur for the three looks, sensor equivalence and EXIF reading |
| `pre-eye-target-sensor` | Adds `--target-sensor`, so a look can be rendered for another sensor size |
| `v0.1.0` | Adds eye and face detection, a focus zone, depth-edge refinement, grain matching and a first whisker mask |
| `v0.1.1` | Scales blur strength with focal length |
| `v0.3.0-physical` | Physical-optics mode with metric depth models, `--blur-scale` and `--depth-weights` |

`main` is the current version and includes physical mode. Eye and face detection live in the `v0.1.x` tags and are not
part of `main`.

## Licences

This code is licensed under the GNU Affero General Public License, version 3 only (AGPL-3.0-only). See the [LICENSE](LICENSE) file.
Copyright (C) 2026 Qavats Ranch, LLC.

You may use, modify and share it freely. If you distribute a modified version, or run one as a service that other people use over a network, you must make your source code available to them under the same license.

The depth model weights are not part of this repository and each has its own licence. Check the model card before commercial use.
