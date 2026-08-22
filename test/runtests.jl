using Test
using YOLOWeights
using SHA: sha256

# These tests download from the pinned Ultralytics release URLs (once; the
# scratch space caches across runs). They need network access.

@testset "YOLOWeights" begin
    @testset "registry" begin
        @test YOLOWeights.available() == ["yolo11n", "yolov8n"]
        @test_throws ArgumentError YOLOWeights.model_path("yolov99x")
        for (name, spec) in YOLOWeights.MODELS
            @test occursin(r"^[0-9a-f]{64}$", spec.sha256)
            @test startswith(spec.url, "https://github.com/ultralytics/assets/")
            @test endswith(spec.file, ".onnx")
        end
    end

    @testset "download, verify, cache" begin
        path = model_path("yolov8n")
        @test isfile(path)
        @test open(io -> bytes2hex(sha256(io)), path) ==
              YOLOWeights.MODELS["yolov8n"].sha256

        # Second call is a cache hit: same path, file untouched.
        mt = mtime(path)
        @test model_path("yolov8n") == path
        @test mtime(path) == mt

        # A corrupted cache entry is discarded and re-fetched, not returned.
        write(path, "not an onnx file")
        path2 = model_path("yolov8n")
        @test path2 == path
        @test open(io -> bytes2hex(sha256(io)), path2) ==
              YOLOWeights.MODELS["yolov8n"].sha256
    end
end
