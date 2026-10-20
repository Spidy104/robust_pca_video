using LinearAlgebra
using Printf
using Random
using RobustPCA

const DATA = joinpath(dirname(@__DIR__), "data")
const OUT = joinpath(dirname(@__DIR__), "outputs")
const WINDOW = 200
const STEP, BINS = 0.005, 100
const CALIBRATION = "highway"
const METHODS = (:batch_w5, :batch_w20, :online, :median)
const WIDTHS = (batch_w5=5, batch_w20=20)
const LABELS = (batch_w5="batch RPCA w=5", batch_w20="batch RPCA w=20", online="online RPCA", median="temporal median")
const VIDEOS = ("highway" => "none", "backdoor" => "shadows", "boats" => "rippling water",
                "badminton" => "camera shake", "abandonedBox" => "objects that stop")
const BATCH = PCPOptions(tol=1e-3)
const ONLINE = OnlineOptions()
const SWEEP = ((0.99, 1.0), (0.995, 1.0), (0.999, 1.0), (1.0, 1.0), (0.999, 0.5), (0.999, 2.0), (0.999, 4.0))

relative(A, B) = norm(A - B) / norm(B)
counts_path(video) = joinpath(OUT, "$(video)_counts.csv")
new_counts() = PRCounts(STEP, BINS)
windows(N) = ((lo = min(start, N - WINDOW + 1); (lo:lo + WINDOW - 1, (start - lo + 1):WINDOW)) for start in 1:WINDOW:N)
function load_clip(video)
    isdir(joinpath(DATA, video)) || error("no data for $video; run scripts/fetch_cdnet.sh first")
    read_video(joinpath(DATA, video))
end

function phase_grid()
    shape = (4800, 200)
    ws = RPCAWorkspace(shape..., 60)
    path = joinpath(OUT, "synthetic.csv")
    open(path, "w") do io
        println(io, "rank,density,iterations,seconds,error_L,error_S")
        @printf("%5s %8s %6s %8s %10s %10s\n", "rank", "density", "iters", "time", "‖ΔL‖/‖L0‖", "‖ΔS‖/‖S0‖")
        for r in (5, 10, 20, 30, 40), density in (0.02, 0.05, 0.1, 0.2, 0.3)
            L0, S0 = synthetic(Xoshiro(r), shape, r, density)
            X = L0 + S0
            seconds = @elapsed result = rpca!(ws, X)
            row = (r, density, result.iterations, seconds, relative(ws.L, L0), relative(ws.S, S0))
            println(io, join(row, ","))
            @printf("%5d %8.2f %6d %7.2fs %10.1e %10.1e\n", row...)
        end
    end
    println("wrote ", path)
end

function save_counts(path, counts)
    open(path, "w") do io
        println(io, "threshold,", join(("$(k)_positives,$(k)_negatives" for k in keys(counts)), ","))
        grid = thresholds(first(counts))
        for k in eachindex(first(counts).positives)
            threshold = k <= length(grid) ? @sprintf("%.3f", grid[k]) : "Inf"
            println(io, threshold, ",", join(("$(c.positives[k]),$(c.negatives[k])" for c in counts), ","))
        end
    end
end

function load_curves(path)
    lines = collect(eachline(path))
    names = Tuple(Symbol.(replace.(split(lines[1], ",")[2:2:end], "_positives" => "")))
    rows = [split(line, ",")[2:end] for line in lines[2:end]]
    column(j) = parse.(Int, getindex.(rows, j))
    curves = (pr_curve(PRCounts(STEP, column(2i - 1), column(2i))) for i in eachindex(names))
    NamedTuple{names}(Tuple(curves))
end

function calibrate()
    isfile(counts_path(CALIBRATION)) || error("run `cdnet $CALIBRATION` first: it calibrates the thresholds")
    map(curve -> argmax(curve.f1), load_curves(counts_path(CALIBRATION)))
end

function evaluate(video)
    bytes, gt, dims = load_clip(video)
    m, N = size(bytes)
    N >= WINDOW || error("$video has $N frames, fewer than one window")
    X = Matrix{Float64}(undef, m, WINDOW)
    batch = map(width -> RPCAWorkspace(m, WINDOW, width), WIDTHS)
    counts = NamedTuple{METHODS}(ntuple(_ -> new_counts(), length(METHODS)))
    smoother = Smoother(dims)
    online = nothing
    background, column, residual, s = zeros(m), zeros(WINDOW), zeros(m), zeros(m)
    seconds = zeros(3)
    for (cols, fresh) in windows(N)
        X .= view(bytes, :, cols) ./ 255
        seconds[1] += @elapsed rpca!(batch.batch_w5, X, BATCH)
        seconds[2] += @elapsed rpca!(batch.batch_w20, X, BATCH)
        isnothing(online) && (online = OnlineRPCA(batch.batch_w5, ONLINE))
        median_background!(background, X, column)
        for j in fresh
            truth = view(gt, :, cols[j])
            for method in keys(batch)
                tally!(counts[method], smooth!(smoother, view(batch[method].S, :, j)), truth)
            end
            seconds[3] += @elapsed step!(online, s, view(X, :, j))
            tally!(counts.online, smooth!(smoother, s), truth)
            residual .= view(X, :, j) .- background
            tally!(counts.median, smooth!(smoother, residual), truth)
        end
    end
    @printf("%s: %d frames of %d×%d; frames/s: batch w=5 %.0f, batch w=20 %.0f, online %.0f\n",
            video, N, dims..., (N ./ seconds)...)
    save_counts(counts_path(video), counts)
end

function summary()
    taus = calibrate()
    table = IOBuffer()
    println(table, "| video | stress | ", join((LABELS[k] for k in METHODS), " | "), " |")
    println(table, "|---|---|", "---:|"^length(METHODS))
    for (video, stress) in VIDEOS
        isfile(counts_path(video)) || continue
        curves = load_curves(counts_path(video))
        cells = (@sprintf("%.3f / **%.3f** (P %.2f, R %.2f)", maximum(curves[k].f1), curves[k].f1[taus[k]],
                          curves[k].precision[taus[k]], curves[k].recall[taus[k]]) for k in METHODS)
        println(table, "| ", video == CALIBRATION ? "$video (calibration)" : video, " | ", stress, " | ", join(cells, " | "), " |")
    end
    println(table, "\nCells: oracle F1 (best τ for that video) / **F1 at the τ calibrated on $CALIBRATION** (precision, recall).")
    println(table, "Calibrated τ: ", join((@sprintf("%s %.3f", LABELS[k], STEP * taus[k]) for k in METHODS), ", "), ".")
    text = String(take!(table))
    print(text)
    write(joinpath(OUT, "summary.md"), text)
    println("wrote outputs/summary.md")
end

function sweep_video(video)
    bytes, gt, dims = load_clip(video)
    m, N = size(bytes)
    first_window = bytes[:, 1:WINDOW] ./ 255
    ws = RPCAWorkspace(first_window, WIDTHS.batch_w5)
    rpca!(ws, first_window, BATCH)
    models = [OnlineRPCA(ws, OnlineOptions(γ=γ, λ₂=factor)) for (γ, factor) in SWEEP]
    counts = [new_counts() for _ in SWEEP]
    smoother = Smoother(dims)
    x, s = zeros(m), zeros(m)
    seconds = @elapsed for j in 1:N
        x .= view(bytes, :, j) ./ 255
        for (model, histogram) in zip(models, counts)
            step!(model, s, x)
            tally!(histogram, smooth!(smoother, s), view(gt, :, j))
        end
    end
    @printf("%s: %d frames × %d settings in %.1f s\n", video, N, length(SWEEP), seconds)
    pr_curve.(counts)
end

function online_sweep()
    curves = Dict(video => sweep_video(video) for (video, _) in VIDEOS)
    taus = argmax.(getproperty.(curves[CALIBRATION], :f1))
    table = IOBuffer()
    println(table, "| γ | λ₂/λ₀ | ", join(first.(VIDEOS), " | "), " |")
    println(table, "|---:|---:|", "---:|"^length(VIDEOS))
    for (i, (γ, factor)) in enumerate(SWEEP)
        cells = (@sprintf("%.3f (%.3f)", curves[video][i].f1[taus[i]], maximum(curves[video][i].f1)) for (video, _) in VIDEOS)
        println(table, "| $γ | $factor | ", join(cells, " | "), " |")
    end
    println(table, "\nCells: F1 at the τ calibrated on $CALIBRATION (oracle F1). Default: γ = $(ONLINE.γ), λ₂/λ₀ = $(ONLINE.λ₂).")
    text = String(take!(table))
    print(text)
    write(joinpath(OUT, "online_sweep.md"), text)
end

function threshold_mask(S, smoother, τ)
    mask = falses(size(S))
    for j in axes(S, 2)
        mask[:, j] .= smooth!(smoother, view(S, :, j)) .> τ
    end
    mask
end

function mosaic(path, rows, gt, dims)
    w, h = dims
    frame = Matrix{UInt8}(undef, 5w, h * length(rows))
    gray(x) = round(UInt8, clamp(x, 0, 1) * 255)
    label(g) = g == 0xff ? 0xff : g == 0x32 ? 0x80 : g == 0x00 ? 0x00 : 0x30
    codec = endswith(path, ".mp4") ? `-c:v libx264 -crf 26 -pix_fmt yuv420p` : ``
    open(`ffmpeg -v error -y -f rawvideo -pix_fmt gray -s $(5w)x$(h * length(rows)) -r 15 -i - -vf 'scale=iw*2:ih*2:flags=neighbor' $codec $path`, "w") do io
        for j in axes(gt, 2)
            for (row, r) in enumerate(rows), y in 1:h, x in 1:w
                p, Y = (y - 1) * w + x, (row - 1) * h + y
                frame[x, Y] = gray(r.input[p, j])
                frame[w + x, Y] = gray(r.background[p, j])
                frame[2w + x, Y] = gray(4abs(r.foreground[p, j]))
                frame[3w + x, Y] = r.mask[p, j] ? 0xff : 0x00
                frame[4w + x, Y] = label(gt[p, j])
            end
            write(io, frame)
        end
    end
end

column_slice(rows, j) = map(r -> map(M -> view(M, :, j:j), r), rows)

function replay_online(bytes, ws, cols)
    m = size(bytes, 1)
    online = OnlineRPCA(ws, ONLINE)
    L, S = zeros(m, length(cols)), zeros(m, length(cols))
    x, s = zeros(m), zeros(m)
    for j in 1:last(cols)
        x .= view(bytes, :, j) ./ 255
        step!(online, s, x)
        j in cols || continue
        S[:, j - first(cols) + 1] .= s
        L[:, j - first(cols) + 1] .= online.background
    end
    L, S
end

function render(video, center)
    bytes, gt, dims = load_clip(video)
    m, N = size(bytes)
    isnothing(center) || 1 <= center <= N || error("frame $center is outside 1:$N")
    busiest = isnothing(center) ? argmax(count(==(0xff), view(gt, :, j)) for j in 1:N) : center
    name = isnothing(center) ? video : "$(video)_$center"
    lo = clamp(busiest - WINDOW ÷ 2, 1, N - WINDOW + 1)
    cols = lo:lo + WINDOW - 1
    taus = calibrate()
    smoother = Smoother(dims)

    ws = RPCAWorkspace(m, WINDOW, WIDTHS.batch_w5)
    rpca!(ws, bytes[:, 1:WINDOW] ./ 255, BATCH)
    L, S = replay_online(bytes, ws, cols)
    X = bytes[:, cols] ./ 255
    rpca!(ws, X, BATCH)
    rows = ((input=X, background=ws.L, foreground=ws.S, mask=threshold_mask(ws.S, smoother, STEP * taus.batch_w5)),
            (input=X, background=L, foreground=S, mask=threshold_mask(S, smoother, STEP * taus.online)))
    truth = gt[:, cols]
    mosaic(joinpath(OUT, "$name.mp4"), rows, truth, dims)
    j = busiest - lo + 1
    mosaic(joinpath(OUT, "$name.png"), column_slice(rows, j), truth[:, j:j], dims)
    println("  wrote outputs/$name.mp4 (ROI frames $(first(cols))–$(last(cols))) and $name.png")
end

function bench_row(X, spec, f1_of)
    name, width, exact = spec
    ws = RPCAWorkspace(X, width)
    rpca!(ws, X, PCPOptions(tol=1e-3, exact=exact, max_iter=2))
    runs = [@timed(rpca!(ws, X, PCPOptions(tol=1e-3, exact=exact))) for _ in 1:(exact ? 1 : 3)]
    seconds = sort([r.time for r in runs])[cld(length(runs), 2)]
    (; name, seconds, bytes=runs[1].bytes, result=runs[1].value, f1=f1_of(ws.S), L=ws.L)
end

function bench(video)
    BLAS.set_num_threads(1)
    bytes, labels, dims = load_clip(video)
    X, gt = bytes[:, 1:WINDOW] ./ 255, labels[:, 1:WINDOW]
    smoother = Smoother(dims)
    function best_f1(S)
        counts = new_counts()
        for j in axes(S, 2)
            tally!(counts, smooth!(smoother, view(S, :, j)), view(gt, :, j))
        end
        maximum(pr_curve(counts).f1)
    end
    report = IOBuffer()
    line(s) = (println(s); println(report, s))
    line(@sprintf("%s, %d×%d, BLAS threads = 1, randomized: median of 3 warm runs, exact: 1 warm run", video, size(X)...))
    line(@sprintf("%-16s %8s %6s %5s %10s %12s %7s %14s", "SVT", "time", "iters", "rank", "alloc", "alloc/iter", "F1", "‖L-L_exact‖/‖L‖"))
    reference = nothing
    for spec in (("exact GR", 40, true), ("randomized w=20", 20, false), ("randomized w=40", 40, false), ("randomized w=80", 80, false))
        row = bench_row(X, spec, best_f1)
        reference = something(reference, copy(row.L))
        line(@sprintf("%-16s %7.2fs %6d %5d %8.1fMB %10.2fMB %7.3f %14.1e", row.name, row.seconds, row.result.iterations, row.result.rank,
                      row.bytes / 2^20, row.bytes / row.result.iterations / 2^20, row.f1, relative(row.L, reference)))
    end
    ws = RPCAWorkspace(X, 40)
    rpca!(ws, X, BATCH)
    live = map(1:5) do _
        rpca!(ws, X, BATCH)
        GC.gc()
        Base.gc_live_bytes() / 2^20
    end
    line(@sprintf("live heap after each of 5 solves: %s MB", join((@sprintf("%.1f", x) for x in live), ", ")))
    write(joinpath(OUT, "bench.txt"), take!(report))
    println("wrote outputs/bench.txt")
end

command = get(ARGS, 1, "synthetic")
if command == "synthetic"
    phase_grid()
elseif command == "cdnet"
    evaluate(ARGS[2])
    isfile(counts_path(CALIBRATION)) && render(ARGS[2], nothing)
elseif command == "render"
    render(ARGS[2], length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing)
elseif command == "summary"
    summary()
elseif command == "sweep"
    online_sweep()
elseif command == "bench"
    bench(get(ARGS, 2, "highway"))
else
    error("usage: run.jl synthetic | cdnet <video> | render <video> [frame] | summary | sweep | bench [video]")
end
