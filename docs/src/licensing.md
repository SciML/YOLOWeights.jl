# [Weight licensing](@id weight-licensing)

Read this before shipping anything.

The **package code** is MIT. The licenses of the **weights** differ by family, and
[`YOLOWeights.license`](@ref) tells you which applies to a given name:

```julia
YOLOWeights.license("yolov8n")     # :AGPL3
YOLOWeights.license("yolox_s")     # :Apache2
```

## The two licenses

**Ultralytics families — `yolov8`, `yolo11`, `yolo26` — are AGPL-3.0**, © Ultralytics.
Internal development and testing is fine. Shipping them inside a proprietary product
triggers the AGPL's obligations, and in practice needs an
[Ultralytics commercial license](https://www.ultralytics.com/license).

**YOLOX is Apache-2.0**, © Megvii. No copyleft obligations; safe for proprietary
products under the usual Apache terms.

## Selecting on license

If you need permissively-licensed weights, filter for them rather than relying on
a name convention:

```julia
YOLOWeights.available(license = :Apache2)
```

and assert it at the point of use if the distinction matters to your build:

```julia
name = "yolox_s"
YOLOWeights.license(name) === :Apache2 || error("$name is not permissively licensed")
path = model_path(name)
```

## What this package does and does not change

Keeping AGPL content behind an optional dependency is the whole reason this package
exists as a separate unit. To be precise about what that buys:

- It keeps AGPL-licensed weights out of *other packages'* distributions. A package
  that depends on YOLOWeights.jl — or weak-depends on it — does not itself
  redistribute the weights.
- It does **not** change what the AGPL requires of *you* once you obtain and use
  those weights. The obligations attach to your use and distribution, not to how
  the bytes reached your machine.

Because the weights are downloaded from upstream at use time rather than mirrored
into artifacts, this package is not the distributor. See
[Why not `Artifacts.toml`?](@ref) for the mechanics.
