# YOLOWeights.jl

Pinned, checksummed access to the major pregenerated YOLO detector weights in ONNX
form — the Ultralytics YOLOv8 / YOLO11 / YOLO26 families and the Apache-2.0-licensed
YOLOX family.

```julia
using YOLOWeights

path = model_path("yolov8n")                                # downloads + verifies on first use
YOLOWeights.available()                                     # every model name, sorted
YOLOWeights.available(task = :detect, license = :Apache2)   # the YOLOX set
YOLOWeights.spec("yolo26n")                                 # task, family, license, input, output layout
YOLOWeights.license("yolov8n")                              # :AGPL3 — check before shipping
```

## Installation

```julia
using Pkg
Pkg.add("YOLOWeights")
```

## How the weights are obtained

The weights are **not** shipped inside this package. [`model_path`](@ref) fetches
them from the official upstream release assets into a per-package scratch space,
verifies the SHA-256 against the pin recorded in `src/YOLOWeights.jl`, and returns
the local path.

The verification rules are deliberately strict:

- A cached file that verifies is never re-fetched.
- A cached file that fails verification is discarded and re-fetched once.
- A *fresh* download that fails verification is an error — upstream changed the
  asset, and the package will not hand you bytes it cannot vouch for.

Downloads land under a temporary `.part` name and are moved into place only after
verification, so an interrupted download can never masquerade as a verified file.

Every hash in the registry was computed from a clean download of the official
asset, and every output shape documented was read from the ONNX graph headers of
those exact files rather than copied from a model card.

## Licensing, briefly

The **package code** is MIT. The **weights** are not uniformly licensed: the
Ultralytics families are AGPL-3.0 and the YOLOX family is Apache-2.0. This matters
before you ship anything, so it has [its own page](@ref weight-licensing) — and
[`YOLOWeights.license`](@ref) will tell you which applies to a given name at
runtime.

## Why not `Artifacts.toml`?

Julia artifacts must be content-addressed tarballs, and upstream publishes bare
`.onnx` files. Mirroring them into our own tarballs would make this package the
*distributor* of the weights — which, for the AGPL families, is precisely what it
is designed to avoid. Downloading from upstream at use time, pinned by URL and
SHA-256, leaves distribution where it already is.

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices.
- There are a few community forums for getting help and asking questions:
  the #diffeq-bridged and #sciml-bridged channels in the
  [Julia Slack](https://julialang.org/slack/), the #diffeq-bridged and #sciml-bridged
  channels in the [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged),
  on the [Julia Discourse forums](https://discourse.julialang.org), or
  see also [SciML Community page](https://sciml.ai/community/).
