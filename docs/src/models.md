# Model registry

Every name below is accepted by [`model_path`](@ref). The tables record what the
raw output tensors mean, because that is the part you cannot recover from the file
itself without reading the graph.

Sizes are the on-disk size of the ONNX file. Input is the square input side in
pixels; a dynamic-axes export reports `0` from
[`YOLOWeights.spec`](@ref).

## Ultralytics detection — 80-class COCO, AGPL-3.0

All from [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0),
input 640×640 unless noted.

YOLOv8 and YOLO11 output `[1, 84, 8400]` channels-first, and need a max-over-class
step followed by NMS. **YOLO26 is end-to-end**: it emits `[1, 300, 6]` final rows
and requires no NMS.

| Family | Sizes | Notes |
|---|---|---|
| YOLOv8 | `yolov8n` (13 MB) · `yolov8s` (43 MB) · `yolov8m` (100 MB) · `yolov8l` (167 MB) · `yolov8x` (261 MB) | |
| YOLO11 | `yolo11n` (11 MB) · `yolo11s` (37 MB) · `yolo11m` (77 MB) · `yolo11l` (98 MB) · `yolo11x` (218 MB) | `yolo11n` is a dynamic-axes export (free batch/height/width) |
| YOLO26 | `yolo26n` (10 MB) · `yolo26s` (37 MB) · `yolo26m` (79 MB) · `yolo26l` (95 MB) · `yolo26x` (213 MB) | NMS-free |

## Ultralytics task variants — nano size, AGPL-3.0

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
| `yolo26n-depth` | `:depth` | 768 | — | monocular depth map `[1, 1, 768, 768]`, **metres**, near = small, clamped to 0.0151–122.25 m |
| `yolo26n-sem` | `:semantic` | 1024 | — | per-pixel map `[1, 1024, 1024]` |
| `yolo26n-reid` | `:reid` | dynamic | — | 512-d appearance embeddings |

Note the input-size irregularity: `yolov8n-cls` takes 640, while `yolo11n-cls` and
`yolo26n-cls` take 224. Read `spec(name).input` rather than assuming.

## YOLOX — 80-class COCO, Apache-2.0

From the official
[Megvii-BaseDetection/YOLOX 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)
release, the only YOLOX release carrying a full official ONNX set. **If the AGPL is
a problem for what you are building, this is the family to use.**

Output is `[1, A, 85]` channels-last (4 box + objectness + 80 classes) — a raw
per-anchor head. Apply the grid/stride decode from the YOLOX ONNXRuntime demo
before NMS.

| Name | Input | Size |
|---|---|---|
| `yolox_nano` | 416 | 4 MB |
| `yolox_tiny` | 416 | 20 MB |
| `yolox_s` | 640 | 35 MB |
| `yolox_m` | 640 | 97 MB |
| `yolox_l` | 640 | 207 MB |
| `yolox_x` | 640 | 378 MB |
| `yolox_darknet` | 640 | 244 MB — the YOLOv3-style Darknet-53 backbone |

## Querying the registry programmatically

The tables above are a convenience; the registry itself is the source of truth and
is queryable at runtime.

```julia
YOLOWeights.available(family = :yolo26)                   # every YOLO26 entry
YOLOWeights.available(task = :segment)                    # across all families
YOLOWeights.available(license = :Apache2)                 # the permissive set

s = YOLOWeights.spec("yolo26n-depth")
s.task        # :depth
s.input       # 768
s.classes     # 0 — the concept does not apply
s.note        # what the output tensor means
```

## Adding a model

Add an entry to `MODELS` via the `_ult` / `_yolox` constructors, or a full
[`YOLOWeights.ModelSpec`](@ref) for a new family, with:

- a SHA-256 computed from **your own clean download** of the official asset —
  never a hash copied from elsewhere;
- `task`, `family` and `license`;
- `input` (`0` for dynamic-axes exports) and `classes` (`0` where the concept does
  not apply);
- a `note` describing the raw output layout, ideally read from the ONNX graph
  itself, and saying "unverified" where it could not be checked.

Keep the tables on this page in sync. The registry-hygiene tests in
`test/runtests.jl` will hold you to most of the rest.
