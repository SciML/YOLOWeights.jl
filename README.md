# YOLOWeights.jl

Pinned, checksummed access to the major pregenerated YOLO detector weights in
ONNX form — the Ultralytics YOLOv8 / YOLO11 / YOLO26 families and the
Apache-2.0-licensed YOLOX family.

```julia
using YOLOWeights
path = model_path("yolov8n")                 # downloads + SHA-256 verifies on first use
YOLOWeights.available()                      # every model name, sorted
YOLOWeights.available(task = :detect, license = :Apache2)   # the YOLOX set
YOLOWeights.spec("yolo26n")                  # task, family, license, input, output layout
YOLOWeights.license("yolov8n")               # :AGPL3 -- check before shipping
```

The weights are **not** shipped inside this package. `model_path` fetches them
from the official upstream release assets into a per-package scratch space,
verifies the SHA-256 against the pin recorded in `src/YOLOWeights.jl`, and
returns the local path. A cached file that verifies is never re-fetched; one
that fails verification is discarded and re-fetched once; a fresh download
that fails verification is an error — upstream changed, do not use it.

Every hash was computed from a clean download of the official asset, and every
output shape quoted below was read from the ONNX graph headers of those exact
files — nothing is copied from model cards.

## Models

### Ultralytics detection — 80-class COCO, AGPL-3.0

All from [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Input 640×640 unless noted. v8/11 output `[1, 84, 8400]` channels-first and
need max-over-class + NMS; **YOLO26 is end-to-end**: `[1, 300, 6]` final rows,
no NMS required.

| Family | Sizes | Notes |
|---|---|---|
| YOLOv8 | `yolov8n` (13 MB) · `yolov8s` (43 MB) · `yolov8m` (100 MB) · `yolov8l` (167 MB) · `yolov8x` (261 MB) | |
| YOLO11 | `yolo11n` (11 MB) · `yolo11s` (37 MB) · `yolo11m` (77 MB) · `yolo11l` (98 MB) · `yolo11x` (218 MB) | `yolo11n` is a dynamic-axes export (free batch/height/width) |
| YOLO26 | `yolo26n` (10 MB) · `yolo26s` (37 MB) · `yolo26m` (79 MB) · `yolo26l` (95 MB) · `yolo26x` (213 MB) | NMS-free |

### Ultralytics task variants — nano size, AGPL-3.0

| Name | Task | Input | Classes | Output |
|---|---|---|---|---|
| `yolov8n-seg`, `yolo11n-seg` | `:segment` | 640 | 80 | `[1, 116, 8400]` + protos `[1, 32, 160, 160]` |
| `yolo26n-seg` | `:segment` | 640 | 80 | NMS-free `[1, 300, 38]` + protos |
| `yolov8n-pose`, `yolo11n-pose` | `:pose` | 640 | 1 (person) | `[1, 56, 8400]`: box + score + 17×(x, y, conf) |
| `yolo26n-pose` | `:pose` | 640 | 1 (person) | NMS-free `[1, 300, 57]` |
| `yolov8n-obb`, `yolo11n-obb` | `:obb` | 1024 | 15 (DOTA) | `[1, 20, 21504]`: box + classes + angle |
| `yolo26n-obb` | `:obb` | 1024 | 15 (DOTA) | NMS-free `[1, 300, 7]` |
| `yolov8n-cls` | `:classify` | **640** | 1000 | `[1, 1000]` ImageNet scores |
| `yolo11n-cls`, `yolo26n-cls` | `:classify` | 224 | 1000 | `[1, 1000]` ImageNet scores |
| `yolo26n-depth` | `:depth` | 768 | — | monocular depth map `[1, 1, 768, 768]` |
| `yolo26n-sem` | `:semantic` | 1024 | — | per-pixel map `[1, 1024, 1024]` |
| `yolo26n-reid` | `:reid` | dynamic | — | 512-d appearance embeddings |

### YOLOX — 80-class COCO, **Apache-2.0 (permissively licensed)**

From the official [Megvii-BaseDetection/YOLOX 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)
release — the only YOLOX release with a full official ONNX set. **If the AGPL
is a problem for what you're building, this is the family to use.** Output is
`[1, A, 85]` channels-last (4 box + objectness + 80 classes), a raw per-anchor
head: apply the grid/stride decode from the YOLOX ONNXRuntime demo before NMS.

| Name | Input | Size |
|---|---|---|
| `yolox_nano` | 416 | 4 MB |
| `yolox_tiny` | 416 | 20 MB |
| `yolox_s` | 640 | 35 MB |
| `yolox_m` | 640 | 97 MB |
| `yolox_l` | 640 | 207 MB |
| `yolox_x` | 640 | 378 MB |
| `yolox_darknet` | 640 | 244 MB — the YOLOv3-style Darknet-53 backbone |

## License — read this before shipping anything

The **package code** is MIT. The licenses of the **weights** differ by family,
and `YOLOWeights.license(name)` tells you which applies:

- **Ultralytics families (yolov8/yolo11/yolo26): AGPL-3.0**, © Ultralytics.
  Internal development and testing is fine; shipping them inside a proprietary
  product triggers the AGPL's obligations and in practice needs an
  [Ultralytics commercial license](https://www.ultralytics.com/license).
- **YOLOX: Apache-2.0**, © Megvii. No copyleft obligations; safe for
  proprietary products under the usual Apache terms.

Keeping AGPL content behind this optional dependency is the whole reason the
package exists as a separate unit: nothing here changes what the AGPL requires
of *you*; the package only keeps the AGPL content out of *other packages'*
distributions.

## Why not Artifacts.toml?

Julia artifacts must be content-addressed tarballs; upstream publishes bare
`.onnx` files, and mirroring them into our own tarballs would make *us* the
distributor — which for the AGPL families is exactly what this package
avoids. Downloading from upstream at use time, pinned by URL and SHA-256,
keeps distribution where it already is.

## Testing

`Pkg.test()` checks registry hygiene for every entry and downloads only the
smallest model of each family (`yolov8n`, `yolo11n`, `yolo26n`, `yolox_nano`).
Set `ENV["YOLOWEIGHTS_TEST_ALL"] = "true"` to download and SHA-verify **every**
pin — several GB of traffic; the sweep deletes what it fetched afterwards.

## Adding a model

Add an entry to `MODELS` via the `_ult`/`_yolox` constructors (or a
`ModelSpec` for a new family) with: a SHA-256 computed from your own clean
download of the official asset — never a hash copied from elsewhere — plus
`task`, `family`, `license`, `input` (0 for dynamic exports), `classes`
(0 where the concept doesn't apply), and a `note` describing the raw output
layout, ideally read from the ONNX graph itself, saying "unverified" where it
wasn't. Keep the README tables in sync; the registry-hygiene tests will hold
you to most of it.
