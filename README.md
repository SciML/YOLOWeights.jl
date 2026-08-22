# YOLOWeights.jl

Pinned, checksummed access to pregenerated [Ultralytics](https://github.com/ultralytics)
YOLO detector weights in ONNX form.

```julia
using YOLOWeights
path = model_path("yolov8n")     # downloads + SHA-256 verifies on first use
YOLOWeights.available()          # ["yolo11n", "yolov8n"]
```

The weights are **not** shipped inside this package. `model_path` fetches them
from the official Ultralytics release assets into a per-package scratch space,
verifies the SHA-256 against the pin recorded in `src/YOLOWeights.jl`, and
returns the local path. A cached file that verifies is never re-fetched; one
that fails verification is discarded and re-fetched once; a fresh download
that fails verification is an error — upstream changed, do not use it.

## Models

| Name | Task | Input | Source |
|---|---|---|---|
| `yolov8n` | 80-class COCO detection | 1x3x640x640, RGB, /255 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |
| `yolo11n` | 80-class COCO detection | 1x3x640x640, RGB, /255 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) |

## License — read this before shipping anything

The **package code** is MIT. The **weights it downloads are AGPL-3.0**,
© Ultralytics — that is the whole reason this package exists as a separate
unit: so that packages under other licenses can reach the weights through an
optional dependency instead of vendoring AGPL content.

Using these weights during internal development and testing is fine. Shipping
them inside a proprietary product triggers the AGPL's obligations and in
practice needs an [Ultralytics commercial license](https://www.ultralytics.com/license).
Nothing here changes what the AGPL requires of *you*; the package only keeps
the AGPL content out of *other packages'* distributions. If the license ever
becomes a problem, an in-house-trained or permissively-licensed model behind
the same `model_path` interface replaces these.

## Why not Artifacts.toml?

Julia artifacts must be content-addressed tarballs; Ultralytics publishes bare
`.onnx` files, and mirroring them into our own tarballs would make *us* the
AGPL distributor. Downloading from upstream at use time, pinned by URL and
SHA-256, keeps distribution where it already is.

## Adding a model

Add a `ModelSpec` to `MODELS` with a SHA-256 computed from your own clean
download of the official asset, never a hash copied from elsewhere. Keep the
table in the README in sync.
