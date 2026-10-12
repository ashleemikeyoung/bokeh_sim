# bokeh-sim

Re-render a finished photo as Summicron f/2, Summilux f/1.4 and Noctilux f/0.95 on full frame,
using Depth Anything V2 for depth.

    python bokeh_sim.py photo.jpg --lenses noctilux
    python bokeh_sim.py photo.jpg --focus-zone 0.12 --strength 0.03

Run `python bokeh_sim.py --help` for all options. `python bokeh_sim.py --version` shows the version.

Versions are git tags (`git tag`); experimental work lives on branches.

## Changelog

- **v0.1.1**: blur strength now scales with full-frame-equivalent focal length (`--strength` applies at
  35mm-equivalent; a 150mm-equivalent shot gets about 4x). Fixes backgrounds that stayed too sharp on
  telephoto shots. `--no-focal-scale` turns it off.
- **v0.1.0**: baseline (depth, eye focus, focus zone, grain, whisker mask v1).
- Branch `whisker-fix`: unfinished experiment on keeping whiskers; not recommended yet.

## Licences

This code is licensed under the GNU Affero General Public License, version 3 only (AGPL-3.0-only). See the [LICENSE](LICENSE) file.
Copyright (C) 2026 Qavats Ranch, LLC.

You may use, modify and share it freely. If you distribute a modified version, or run one as a service that other people use over a network, you must make your source code available to them under the same license.

The depth model weights are not part of this repository and each has its own licence. Check the model card before commercial use.
