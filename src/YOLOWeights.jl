"""
    YOLOWeights

Pinned, checksummed access to pregenerated Ultralytics YOLO detector weights
in ONNX form. The weights are **not** shipped inside this package: `model_path`
downloads them from the official Ultralytics release assets on first use into
a per-package scratch space, verifies the SHA-256 against the pin recorded
here, and returns the local path. A file that is already present and verifies
is never re-fetched.

This package exists to keep AGPL-3.0 content out of packages under other
licenses: depend on it (or weak-depend on it) instead of vendoring the
weights. **The weights themselves are AGPL-3.0**, © Ultralytics — see the
README before shipping them inside anything proprietary.

```julia
using YOLOWeights
path = YOLOWeights.model_path("yolov8n")   # downloads on first call
YOLOWeights.available()                    # what can be asked for
```
"""
module YOLOWeights

using Downloads: Downloads
using SHA: sha256
using Scratch: @get_scratch!

export model_path

struct ModelSpec
    file::String
    url::String
    sha256::String
end

# Pinned by URL and SHA-256, exactly as downloaded and verified. Sizes are the
# "n" (nano) variants; add heavier variants only with a hash verified from a
# clean download, never copied from a forum post.
const MODELS = Dict{String, ModelSpec}(
    "yolov8n" => ModelSpec(
        "yolov8n.onnx",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx",
        "b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd",
    ),
    "yolo11n" => ModelSpec(
        "yolo11n.onnx",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx",
        "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
    ),
)

"Names `model_path` accepts, sorted."
available() = sort!(collect(keys(MODELS)))

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
verification is an error — upstream changed, do not use it.
"""
function model_path(name::AbstractString)
    spec = get(MODELS, name, nothing)
    spec === nothing && throw(ArgumentError(
        "unknown model $(repr(name)); available: $(join(available(), ", "))"))
    dir = @get_scratch!("models")
    path = joinpath(dir, spec.file)
    _verifies(path, spec.sha256) && return path
    # Absent, or present-but-corrupt: fetch into a temp name so a failed or
    # interrupted download can never masquerade as a verified file.
    part = path * ".part"
    Downloads.download(spec.url, part)
    if !_verifies(part, spec.sha256)
        rm(part; force = true)
        error("checksum mismatch for $(spec.url) -- upstream changed, do not use")
    end
    mv(part, path; force = true)
    return path
end

end # module
