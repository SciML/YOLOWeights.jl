using Documenter, YOLOWeights

makedocs(
    modules = [YOLOWeights],
    sitename = "YOLOWeights.jl",
    clean = true,
    doctest = false,
    linkcheck = true,
    checkdocs = :public,
    linkcheck_ignore = [
        # Release-asset URLs are pinned by SHA-256 rather than reachability, and
        # GitHub rate-limits the crawler across the ~40 of them in the registry.
        r"^https://github\.com/ultralytics/assets/releases/",
        r"^https://github\.com/Megvii-BaseDetection/YOLOX/releases/",
    ],
    format = Documenter.HTML(
        assets = String[],
        canonical = "https://docs.sciml.ai/YOLOWeights/stable/"
    ),
    pages = [
        "index.md",
        "Model registry" => "models.md",
        "Weight licensing" => "licensing.md",
        "API" => "api.md",
    ]
)

deploydocs(repo = "github.com/SciML/YOLOWeights.jl"; push_preview = true)
