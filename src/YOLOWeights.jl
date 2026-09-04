"""
    YOLOWeights

Pinned, checksummed access to the major pregenerated YOLO detector weights in
ONNX form. The weights are **not** shipped inside this package: `model_path`
downloads them from the official upstream release assets on first use into a
per-package scratch space, verifies the SHA-256 against the pin recorded
here, and returns the local path. A file that is already present and verifies
is never re-fetched.

This package exists to keep restrictively-licensed content out of packages
under other licenses: depend on it (or weak-depend on it) instead of
vendoring the weights. **The Ultralytics weights (yolov8/yolo11/yolo26
families) are AGPL-3.0**, © Ultralytics; the YOLOX family is Apache-2.0,
© Megvii — see the README before shipping any of them inside anything
proprietary.

```julia
using YOLOWeights
path = YOLOWeights.model_path("yolov8n")            # downloads on first call
YOLOWeights.available()                             # every model name
YOLOWeights.available(task = :detect, license = :Apache2)
YOLOWeights.spec("yolo26n").note                    # what the tensors mean
```
"""
module YOLOWeights

using Downloads: Downloads
using SHA: sha256
using Scratch: @get_scratch!

export model_path

"""
    ModelSpec

One registry entry. Fields:

- `file`, `url`, `sha256` — the pin: official upstream asset, hashed from a
  clean download of exactly that URL.
- `task::Symbol` — `:detect`, `:segment`, `:pose`, `:obb`, `:classify`,
  `:depth`, `:semantic` or `:reid`.
- `family::Symbol` — `:yolov8`, `:yolo11`, `:yolo26` or `:yolox`.
- `license::Symbol` — the license of the *weights*: `:AGPL3` (Ultralytics)
  or `:Apache2` (Megvii/YOLOX). The package code is MIT either way.
- `input::Int` — square input side in pixels; `0` for a dynamic-axes export.
- `classes::Int` — class count where the concept applies (80 COCO for
  detect/segment, 1000 ImageNet for classify, 15 DOTA for obb, 1 for pose);
  `0` where it does not (depth, semantic, reid).
- `note::String` — the raw output layout. Everything stated as a shape was
  read from the ONNX graph headers of the hashed files themselves;
  interpretations that could not be checked that way say "unverified".
"""
struct ModelSpec
    file::String
    url::String
    sha256::String
    task::Symbol
    family::Symbol
    license::Symbol
    input::Int
    classes::Int
    note::String
end

const ULTRALYTICS_RELEASE =
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
const YOLOX_RELEASE =
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

_ult_family(name) =
    startswith(name, "yolov8") ? :yolov8 :
    startswith(name, "yolo11") ? :yolo11 :
    startswith(name, "yolo26") ? :yolo26 :
    error("not an Ultralytics model name: $name")

# Constructors that keep the registry table below readable. The URL is always
# derived from the name, so a key can never point at a mismatched asset.
_ult(name, sha; task = :detect, input = 640, classes = 80, note) =
    name => ModelSpec(name * ".onnx", ULTRALYTICS_RELEASE * name * ".onnx",
                      sha, task, _ult_family(name), :AGPL3, input, classes, note)
_yolox(name, sha; input = 640, note) =
    name => ModelSpec(name * ".onnx", YOLOX_RELEASE * name * ".onnx",
                      sha, :detect, :yolox, :Apache2, input, 80, note)

# Output-layout notes shared across sizes of a family. All shapes below were
# read from the ONNX graphs of the pinned files (anchor counts are for the
# listed input size).
const _V8_DETECT = "[1, 84, 8400] channels-first (4 box + 80 class scores per \
anchor); needs max-over-class, thresholding and NMS"
const _26_DETECT = "[1, 300, 6] end-to-end, NMS-free: 300 final rows of what \
reads as box + score + class (column order unverified -- check before wiring \
into a fused-style decoder)"
const _YOLOX_DETECT = "[1, A, 85] channels-last (4 box + objectness + 80 class \
scores); raw per-anchor head -- grid/stride decode required before NMS, see \
the YOLOX ONNXRuntime demo"

# Pinned by URL and SHA-256, exactly as downloaded and verified 2026-08-22.
# Never record a hash copied from a model card or forum post: download the
# official asset yourself and hash that.
const MODELS = Dict{String, ModelSpec}(
    # -- Ultralytics detection, 80-class COCO ---------------------------
    _ult("yolov8n", "b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd"; note = _V8_DETECT),
    _ult("yolov8s", "111b9b7df6f1256ec4fa9c9258f10bd824a48da75f7bc575d2f2634c2171ebf7"; note = _V8_DETECT),
    _ult("yolov8m", "3aa21a2bbcb5e374a5802c05c0a68795470dbe67caf6eec15b1802e236692407"; note = _V8_DETECT),
    _ult("yolov8l", "717d57246757ab2e8cdf9770d31987a1ce7fad4165b2820b3d99bc4798c0a75c"; note = _V8_DETECT),
    _ult("yolov8x", "fd6787fbef379d84120e258b09742e8d378ea8fdc40e4b8754e11a905dff2f77"; note = _V8_DETECT),
    _ult("yolo11n", "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b";
         input = 0, note = "as yolov8n, but a dynamic-axes export: input \
[batch, 3, height, width], output [batch, 84, anchors] -- the one dynamic \
model in the Ultralytics set"),
    _ult("yolo11s", "21d6650c5097610c92c76ce5e4b717976059169eaea4962035b90c6a92c07a8f"; note = _V8_DETECT),
    _ult("yolo11m", "8a37b5c53ff642831aa454156b548ec2cf2537827445385c3e1c1b276cb666a3"; note = _V8_DETECT),
    _ult("yolo11l", "8237b045952af25c3c22d8b7424110e86342ea6cf2c4c32d1c096b864bc1b555"; note = _V8_DETECT),
    _ult("yolo11x", "a21552d23ff6eaab86eb6c0e9029da4b00944b75a54c504735f9f4ee38b74561"; note = _V8_DETECT),
    _ult("yolo26n", "2e947b787d9e787b93a16772a5f55b1d4d8c4d86f53146149c5d6a642442d6f7"; note = _26_DETECT),
    _ult("yolo26s", "d26b65c432111eb95798cd2320603d4d75627605dbec6c6b7f98c499a80e7321"; note = _26_DETECT),
    _ult("yolo26m", "5631854916f5d8418169580cde05647f3a1483b21a5026567f122c1fedab973d"; note = _26_DETECT),
    _ult("yolo26l", "d32e8d2b3e5c7c591865d372712e42a88ff5bbc518f3bd83ad3621668de89f7a"; note = _26_DETECT),
    _ult("yolo26x", "88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a"; note = _26_DETECT),

    # -- Ultralytics task variants, nano size ---------------------------
    _ult("yolov8n-seg", "0481e3434424d7a07fff3e6ee4e338e286abf6fe252c1a41ef4a6cacdf2d18f0";
         task = :segment, note = "[1, 116, 8400] (4 box + 80 classes + 32 mask \
coefficients) plus mask protos [1, 32, 160, 160]"),
    _ult("yolov8n-pose", "4abcdec00c4c9891244ffa57a384195ae0d7a56cada809ecde422847e8f669d1";
         task = :pose, classes = 1, note = "[1, 56, 8400]: 4 box + 1 person \
score + 17 keypoints x (x, y, conf)"),
    _ult("yolov8n-obb", "71b1cd4feeeaa50759aea0f77981b119ac10e68ae51862842dece795f1cd4669";
         task = :obb, input = 1024, classes = 15, note = "[1, 20, 21504]: \
4 box + 15 DOTA class scores + rotation angle"),
    _ult("yolov8n-cls", "aa93a10489da776492e9005a9677f2aca3b1db40994e199f54b80131e76ef04d";
         task = :classify, classes = 1000, note = "[1, 1000] ImageNet scores; \
note this one is exported at 640 input, unlike the 224 of yolo11n/yolo26n-cls"),
    _ult("yolo11n-seg", "0bc32bc92e985b881141ef9bd2216e2a746f70519d0d24da9fc85decc4428cf4";
         task = :segment, note = "[1, 116, 8400] (4 box + 80 classes + 32 mask \
coefficients) plus mask protos [1, 32, 160, 160]"),
    _ult("yolo11n-pose", "93e2866b0ce678f99b4dd88af0c12e9ea2edf079a361dc2ecbc6226b77ff6408";
         task = :pose, classes = 1, note = "[1, 56, 8400]: 4 box + 1 person \
score + 17 keypoints x (x, y, conf)"),
    _ult("yolo11n-obb", "848c74ea8a2bd18c426355d536c6d0be567f81ae843740b88a91b153e9012bf9";
         task = :obb, input = 1024, classes = 15, note = "[1, 20, 21504]: \
4 box + 15 DOTA class scores + rotation angle"),
    _ult("yolo11n-cls", "2d2f6414daeced6c518bb13f92b17fc1dbcaad60278b8f3ed5edd341c35abcd4";
         task = :classify, input = 224, classes = 1000, note = "[1, 1000] \
ImageNet scores"),
    _ult("yolo26n-seg", "711e0ef837c677ae4006f65fe503a62a6b0529979f4828ed50b674a7482b7d2e";
         task = :segment, note = "NMS-free: [1, 300, 38] final rows (6 as \
detect + 32 mask coefficients) plus mask protos [1, 32, 160, 160]"),
    _ult("yolo26n-pose", "93fc5e1d6b7690f33b4e1d60d6e9aec1cea14bdbc361bfae11778969be662078";
         task = :pose, classes = 1, note = "NMS-free: [1, 300, 57] final rows \
(6 as detect + 17 keypoints x (x, y, conf))"),
    _ult("yolo26n-obb", "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38";
         task = :obb, input = 1024, classes = 15, note = "NMS-free: \
[1, 300, 7] final rows (6 as detect + rotation angle)"),
    _ult("yolo26n-cls", "4592c7decf302241346fad321d7f155ff796bfc441086426848b0d7175bd20a1";
         task = :classify, input = 224, classes = 1000, note = "[1, 1000] \
ImageNet scores"),
    _ult("yolo26n-depth", "6583ec12521a7e75eacb209db63393158c7e0e772cda9fdad8c8c0fe74af2a1e";
         task = :depth, input = 768, classes = 0, note = "monocular depth map \
[1, 1, 768, 768] in METRES, near = small, clamped by the graph to \
0.0151 .. 122.25 m; the head ends exp(clip(logit, -4, 5) + cal_b) with \
cal_b = -0.19384765625, and the last node is a 4x Resize, so the prediction is \
192 x 192 upsampled"),
    _ult("yolo26n-sem", "7b3881f57103a42a14c40737885346f3349fa0387c9a4ecaa3cc348b3c0c4e9d";
         task = :semantic, input = 1024, classes = 0, note = "per-pixel map \
[1, 1024, 1024]; class vocabulary unverified"),
    _ult("yolo26n-reid", "8529c383197ae4c468eda535d1b165f8b4162cf17bf5fbcff49c7cb6455bc0bb";
         task = :reid, input = 0, classes = 0, note = "dynamic-axes export; \
512-d appearance embeddings [batch, 512] for re-identification"),

    # -- YOLOX, 80-class COCO, Apache-2.0 --------------------------------
    _yolox("yolox_nano", "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d";
           input = 416, note = _YOLOX_DETECT * "; A = 3549 at 416 input"),
    _yolox("yolox_tiny", "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7";
           input = 416, note = _YOLOX_DETECT * "; A = 3549 at 416 input"),
    _yolox("yolox_s", "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063";
           note = _YOLOX_DETECT * "; A = 8400 at 640 input"),
    _yolox("yolox_m", "21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f";
           note = _YOLOX_DETECT * "; A = 8400 at 640 input"),
    _yolox("yolox_l", "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411";
           note = _YOLOX_DETECT * "; A = 8400 at 640 input"),
    _yolox("yolox_x", "c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55";
           note = _YOLOX_DETECT * "; A = 8400 at 640 input"),
    _yolox("yolox_darknet", "10ccc527877c6ee5596929c4b3eee5ad6a78038647260bc9abdfce9a0df55d05";
           note = _YOLOX_DETECT * "; A = 8400 at 640 input; the YOLOv3-style \
Darknet-53 backbone variant"),
)

"""
    available(; task = nothing, family = nothing, license = nothing)

Names `model_path` accepts, sorted. Keywords filter by the corresponding
`ModelSpec` field; `nothing` means "any".

```julia
YOLOWeights.available(task = :detect, license = :Apache2)  # the YOLOX set
YOLOWeights.available(family = :yolo26)
```
"""
function available(; task = nothing, family = nothing, license = nothing)
    names = String[name for (name, s) in MODELS if
                   (task === nothing || s.task === task) &&
                   (family === nothing || s.family === family) &&
                   (license === nothing || s.license === license)]
    return sort!(names)
end

"""
    spec(name::AbstractString) -> ModelSpec

The full registry entry for `name` -- task, family, license, input size,
class count, and what the raw output tensors mean.
"""
function spec(name::AbstractString)
    s = get(MODELS, name, nothing)
    s === nothing && throw(ArgumentError(
        "unknown model $(repr(name)); available: $(join(available(), ", "))"))
    return s
end

"""
    license(name::AbstractString) -> Symbol

License of the *weights* for `name`: `:AGPL3` (Ultralytics) or `:Apache2`
(YOLOX). Check this before wiring a model into anything that ships.
"""
license(name::AbstractString) = spec(name).license

function _verifies(path::AbstractString, hash::AbstractString)
    isfile(path) || return false
    open(path) do io
        bytes2hex(sha256(io)) == hash
    end
end

"""
    model_path(name::AbstractString) -> String

Local path of the ONNX weights for `name` (see [`available`](@ref)),
downloading and SHA-256-verifying them on first use. A cached file that fails
verification is discarded and re-fetched once; a fresh download that fails
verification is an error -- upstream changed, do not use it.
"""
function model_path(name::AbstractString)
    s = spec(name)
    dir = @get_scratch!("models")
    path = joinpath(dir, s.file)
    _verifies(path, s.sha256) && return path
    # Absent, or present-but-corrupt: fetch into a temp name so a failed or
    # interrupted download can never masquerade as a verified file.
    part = path * ".part"
    Downloads.download(s.url, part)
    if !_verifies(part, s.sha256)
        rm(part; force = true)
        error("checksum mismatch for $(s.url) -- upstream changed, do not use")
    end
    mv(part, path; force = true)
    return path
end

end # module
