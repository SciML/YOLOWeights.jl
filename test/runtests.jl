using Test
using YOLOWeights
using YOLOWeights: MODELS, available, spec, license
using SHA: sha256

# The download testsets fetch from the pinned official release URLs (once;
# the scratch space caches across runs) and need network access. By default
# only the smallest model of each family is fetched. Set
# ENV["YOLOWEIGHTS_TEST_ALL"] = "true" to download and verify every pin in
# the registry -- several GB of traffic; the sweep deletes what it fetched
# (except the defaults) so it does not leave the scratch space that large.

sha_of(path) = open(io -> bytes2hex(sha256(io)), path)

const SMALLEST = Dict(:yolov8 => "yolov8n", :yolo11 => "yolo11n",
                      :yolo26 => "yolo26n", :yolox => "yolox_nano")

@testset "YOLOWeights" begin
    @testset "registry hygiene, every entry" begin
        for (name, s) in MODELS
            @test occursin(r"^[0-9a-f]{64}$", s.sha256)
            @test s.file == name * ".onnx"
            @test endswith(s.url, "/" * s.file)
            if s.family === :yolox
                @test startswith(s.url,
                    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/")
                @test s.license === :Apache2
            else
                @test startswith(s.url,
                    "https://github.com/ultralytics/assets/releases/download/")
                @test s.license === :AGPL3
            end
            @test s.task in
                  (:detect, :segment, :pose, :obb, :classify, :depth, :semantic, :reid)
            @test s.family in (:yolov8, :yolo11, :yolo26, :yolox)
            @test s.input >= 0            # 0 = dynamic-axes export
            @test s.classes >= 0          # 0 = concept does not apply
            @test !isempty(s.note)
            # Class counts are structural, not decorative.
            s.task === :detect   && @test s.classes == 80
            s.task === :segment  && @test s.classes == 80
            s.task === :classify && @test s.classes == 1000
            s.task === :obb      && @test s.classes == 15
            s.task === :pose     && @test s.classes == 1
        end
        # One pin is one file: no duplicate URLs or hashes anywhere.
        @test allunique([s.url for s in values(MODELS)])
        @test allunique([s.sha256 for s in values(MODELS)])
    end

    @testset "query API" begin
        @test available() == sort!(collect(keys(MODELS)))
        @test length(available()) == 37

        # Family counts: 5 detect sizes each, plus the nano task variants.
        @test length(available(family = :yolov8)) == 9
        @test length(available(family = :yolo11)) == 9
        @test length(available(family = :yolo26)) == 12
        @test length(available(family = :yolox)) == 7

        # Filters compose, and agree with the registry they filter.
        @test available(task = :detect, license = :Apache2) ==
              available(family = :yolox)
        @test length(available(task = :detect)) == 22
        @test available(task = :depth) == ["yolo26n-depth"]
        @test isempty(available(family = :yolox, task = :segment))
        @test sum(length(available(family = f)) for f in
                  (:yolov8, :yolo11, :yolo26, :yolox)) == length(available())

        @test spec("yolov8n").input == 640
        @test spec("yolo11n").input == 0          # the dynamic-axes export
        @test spec("yolov8n-obb").input == 1024
        @test license("yolov8n") === :AGPL3
        @test license("yolox_s") === :Apache2
        @test_throws ArgumentError spec("yolov99x")
        @test_throws ArgumentError YOLOWeights.model_path("yolov99x")
    end

    @testset "download, verify, cache (smallest of each family)" begin
        for (family, name) in SMALLEST
            path = model_path(name)
            @test isfile(path)
            @test sha_of(path) == MODELS[name].sha256
            # Second call is a cache hit: same path, file untouched.
            mt = mtime(path)
            @test model_path(name) == path
            @test mtime(path) == mt
        end

        # A corrupted cache entry is discarded and re-fetched, not returned.
        path = model_path("yolov8n")
        write(path, "not an onnx file")
        path2 = model_path("yolov8n")
        @test path2 == path
        @test sha_of(path2) == MODELS["yolov8n"].sha256
    end

    if get(ENV, "YOLOWEIGHTS_TEST_ALL", "") == "true"
        @testset "full-registry pin sweep" begin
            keep = Set(values(SMALLEST))
            for name in available()
                path = model_path(name)
                @test sha_of(path) == MODELS[name].sha256
                name in keep || rm(path)   # don't leave GBs in the scratch
            end
        end
    end
end
