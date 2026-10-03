# julia --project=scripts scripts/run.jl synthetic | cdnet <video> | render <video> [frame] | summary | sweep | bench [video]
using LinearAlgebra
using Printf
using Random
using RobustPCA
using Statistics

const DATA = joinpath(dirname(@__DIR__), "data")
const OUT = joinpath(dirname(@__DIR__), "outputs")
const WINDOW = 200                        # frames per RPCA solve
const TOL = 1e-3                          # on video, F1 is unchanged below this and iterations halve (README)
const TAUS = collect(0.005:0.005:0.5)     # threshold on smoothed |residual|, intensity units
const CALIBRATION = "highway"             # the only video whose ground truth picks thresholds
const METHODS = (:batch_w5, :batch_w20, :online, :median)
const LABELS = (batch_w5 = "batch RPCA w=5", batch_w20 = "batch RPCA w=20", online = "online RPCA", median = "temporal median")
const RANK = 5                            # online basis size, = the low batch cap
const GAMMA = 0.999                       # online forgetting factor: ~1000-frame memory, fixed before testing
const VIDEOS = ("highway" => "none", "backdoor" => "shadows", "boats" => "rippling water",
                "badminton" => "camera shake", "abandonedBox" => "objects that stop")

relative(A, B) = norm(A - B) / norm(B)

# Phase transition of exact recovery on video-shaped (tall) problems.
function phase_grid()
    m, n = 4800, 200
    path = joinpath(OUT, "synthetic.csv")
    ws = RPCAWorkspace(m, n; width=60)
    open(path, "w") do io
        println(io, "rank,density,iterations,seconds,error_L,error_S")
        @printf("%5s %8s %6s %8s %10s %10s\n", "rank", "density", "iters", "time", "‖ΔL‖/‖L0‖", "‖ΔS‖/‖S0‖")
        for r in (5, 10, 20, 30, 40), density in (0.02, 0.05, 0.1, 0.2, 0.3)
            L0, S0 = synthetic(m, n, r, density; rng=Xoshiro(r))
            X = L0 + S0
            seconds = @elapsed iterations = rpca!(ws, X)
            eL, eS = relative(ws.L, L0), relative(ws.S, S0)
            @printf("%5d %8.2f %6d %7.2fs %10.1e %10.1e\n", r, density, iterations, seconds, eL, eS)
            println(io, join((r, density, iterations, seconds, eL, eS), ","))
        end
    end
    println("wrote ", path)
end

"""
Per-window buffers for one detector. Its residual is `S` (RPCA) or `X − median` (baseline);
its score is the 3×3 box-smoothed |residual|, tallied over the threshold grid `TAUS`.
"""
struct Detector
    residual::Matrix{Float64}
    score::Matrix{Float64}
    counts::PRCounts
end
Detector(m, n) = Detector(zeros(m, n), zeros(m, n), PRCounts(TAUS))

# Smooth |residual| into `score`; the residual buffer is reused as scratch.
function score!(d::Detector, dims, frame)
    d.residual .= abs.(d.residual)
    box_filter!(d.score, d.residual, dims, 1, frame)
end

# Windows of WINDOW frames tile the video; the last one slides back to stay full length
# and scores only frames no earlier window scored.
windows(N) = ((lo = min(start, N - WINDOW + 1); (lo:lo + WINDOW - 1, (start - lo + 1):WINDOW)) for start in 1:WINDOW:N)

# Score every frame once with each method. Batch RPCA solves each window; online RPCA is
# warm-started from the first window's w=5 solve, then sees each frame once, in order.
function evaluate(video; γ=GAMMA, path=counts_path(video))
    bytes, gt, dims = read_video(joinpath(DATA, video))
    m, N = size(bytes)
    N >= WINDOW || error("$video has $N frames, fewer than one window")
    X = Matrix{Float64}(undef, m, WINDOW)
    batch = (RPCAWorkspace(X; width=5), RPCAWorkspace(X; width=20))
    detectors = NamedTuple{METHODS}(ntuple(_ -> Detector(m, WINDOW), length(METHODS)))
    online = nothing
    background, column, frame = zeros(m), zeros(WINDOW), zeros(m)
    seconds = zeros(3)
    for (cols, new) in windows(N)
        X .= view(bytes, :, cols) ./ 255
        for (k, (ws, d)) in enumerate(zip(batch, (detectors.batch_w5, detectors.batch_w20)))
            seconds[k] += @elapsed rpca!(ws, X; tol=TOL)
            copy!(d.residual, ws.S)
        end
        online = something(online, OnlineRPCA(batch[1], RANK; γ))
        seconds[3] += @elapsed for j in new
            step!(online, view(detectors.online.residual, :, j), view(X, :, j))
        end
        median_background!(background, X, column)
        detectors.median.residual .= X .- background
        for d in detectors
            score!(d, dims, frame)
            tally!(d.counts, view(d.score, :, new), view(gt, :, cols[new]))
        end
    end
    @printf("%s: %d frames of %d×%d; frames/s: batch w=5 %.0f, batch w=20 %.0f, online %.0f\n",
            video, N, dims..., (N ./ seconds)...)
    save_counts(path, detectors)
end

counts_path(video) = joinpath(OUT, "$(video)_counts.csv")

function save_counts(path, detectors)
    open(path, "w") do io
        println(io, "threshold,", join(("$(k)_positives,$(k)_negatives" for k in keys(detectors)), ","))
        for k in 1:length(TAUS) + 1
            t = k <= length(TAUS) ? TAUS[k] : Inf   # last bin: above every threshold
            println(io, t, ",", join(("$(d.counts.positives[k]),$(d.counts.negatives[k])" for d in detectors), ","))
        end
    end
end

# PR curve per method of a saved run, keyed by method.
function load_curves(path)
    lines = collect(eachline(path))
    names = Symbol.(replace.(split(lines[1], ",")[2:2:end], "_positives" => ""))
    rows = [parse.(Float64, split(line, ",")) for line in lines[2:end]]
    column(j) = round.(Int, getindex.(rows, j))
    NamedTuple{Tuple(names)}(Tuple(pr_curve(PRCounts(TAUS, column(2i), column(2i + 1))) for i in eachindex(names)))
end

# Grid index of the best F1 on the calibration video, per method.
calibrate(path=counts_path(CALIBRATION)) = map(curve -> argmax(curve.f1), load_curves(path))

# One τ per method, chosen on the calibration video, applied unchanged to the rest.
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
    println(table, "Calibrated τ: ", join((@sprintf("%s %.3f", LABELS[k], TAUS[taus[k]]) for k in METHODS), ", "), ".")
    text = String(take!(table))
    print(text)
    write(joinpath(OUT, "summary.md"), text)
    println("wrote outputs/summary.md")
end

# Online settings (γ, λ₂/λ₀) around the defaults, with λ₀ = 1/√m: reported, not tuned.
# Each setting gets its own τ calibrated on highway, like every other method.
const SWEEP = ((0.99, 1.0), (0.995, 1.0), (0.999, 1.0), (1.0, 1.0), (0.999, 0.5), (0.999, 2.0), (0.999, 4.0))

function online_sweep()
    curves = Dict{String,Vector}()
    for (video, _) in VIDEOS
        bytes, gt, dims = read_video(joinpath(DATA, video))
        m, N = size(bytes)
        first_window = bytes[:, 1:WINDOW] ./ 255
        ws = RPCAWorkspace(first_window; width=5)
        rpca!(ws, first_window; tol=TOL)
        models = [OnlineRPCA(ws, RANK; γ, λ₂=factor / sqrt(m)) for (γ, factor) in SWEEP]
        counts = [PRCounts(TAUS) for _ in SWEEP]
        x, s, frame = zeros(m), zeros(m), zeros(m)
        residual, score = zeros(m, 1), zeros(m, 1)
        seconds = @elapsed for j in 1:N
            x .= view(bytes, :, j) ./ 255
            for (o, c) in zip(models, counts)
                step!(o, s, x)
                view(residual, :, 1) .= abs.(s)
                box_filter!(score, residual, dims, 1, frame)
                tally!(c, score, view(gt, :, j:j))
            end
        end
        curves[video] = pr_curve.(counts)
        @printf("%s: %d frames × %d settings in %.1f s\n", video, N, length(SWEEP), seconds)
    end
    taus = [argmax(curve.f1) for curve in curves[CALIBRATION]]
    table = IOBuffer()
    println(table, "| γ | λ₂/λ₀ | ", join(first.(VIDEOS), " | "), " |")
    println(table, "|---:|---:|", "---:|"^length(VIDEOS))
    for (i, (γ, factor)) in enumerate(SWEEP)
        cells = (@sprintf("%.3f (%.3f)", curves[video][i].f1[taus[i]], maximum(curves[video][i].f1)) for (video, _) in VIDEOS)
        println(table, "| $γ | $factor | ", join(cells, " | "), " |")
    end
    println(table, "\nCells: F1 at the τ calibrated on $CALIBRATION (oracle F1). Default: γ = $GAMMA, λ₂/λ₀ = 1.")
    text = String(take!(table))
    print(text)
    write(joinpath(OUT, "online_sweep.md"), text)
end

# Stacked rows of input | background L | 4|S| | mask | ground truth, one row per method,
# streamed into ffmpeg.
function mosaic(path, X, rows, gt, (w, h); frames=axes(X, 2))
    frame = Matrix{UInt8}(undef, 5w, h * length(rows))
    byte(x) = round(UInt8, clamp(x, 0, 1) * 255)
    label(g) = g == 0xff ? 0xff : g == 0x32 ? 0x80 : g == 0x00 ? 0x00 : 0x30
    codec = endswith(path, ".mp4") ? `-c:v libx264 -crf 26 -pix_fmt yuv420p` : ``
    open(`ffmpeg -v error -y -f rawvideo -pix_fmt gray -s $(5w)x$(h * length(rows)) -r 15 -i - -vf 'scale=iw*2:ih*2:flags=neighbor' $codec $path`, "w") do io
        for j in frames
            for (row, (L, S, mask)) in enumerate(rows), y in 1:h, x in 1:w
                p, Y = (y - 1) * w + x, (row - 1) * h + y
                frame[x, Y] = byte(X[p, j])
                frame[w + x, Y] = byte(L[p, j])
                frame[2w + x, Y] = byte(4abs(S[p, j]))
                frame[3w + x, Y] = mask[p, j] ? 0xff : 0x00
                frame[4w + x, Y] = label(gt[p, j])
            end
            write(io, frame)
        end
    end
end

# The window around `center` (default: the frame with most labelled motion): batch RPCA
# (w=5) on top, online RPCA below, each masked at its calibrated τ.
function render(video; center=nothing)
    bytes, gt, dims = read_video(joinpath(DATA, video))
    m, N = size(bytes)
    busiest = something(center, argmax(count(==(0xff), view(gt, :, j)) for j in 1:N))
    name = isnothing(center) ? video : "$(video)_$center"
    lo = clamp(busiest - WINDOW ÷ 2, 1, N - WINDOW + 1)
    cols = lo:lo + WINDOW - 1
    taus = calibrate()
    mask(S, method) = (d = Detector(m, WINDOW); copy!(d.residual, S); score!(d, dims, zeros(m)) .> TAUS[taus[method]])

    first_window = bytes[:, 1:WINDOW] ./ 255
    ws = RPCAWorkspace(first_window; width=5)
    rpca!(ws, first_window; tol=TOL)
    online = OnlineRPCA(ws, RANK; γ=GAMMA)
    L, S, x, s = zeros(m, WINDOW), zeros(m, WINDOW), zeros(m), zeros(m)
    for j in 1:last(cols)                    # online has to see every earlier frame
        x .= view(bytes, :, j) ./ 255
        step!(online, s, x)
        if j >= lo
            S[:, j - lo + 1] .= s
            L[:, j - lo + 1] .= online.background
        end
    end
    X = bytes[:, cols] ./ 255
    rpca!(ws, X; tol=TOL)
    rows = ((ws.L, ws.S, mask(ws.S, :batch_w5)), (L, S, mask(S, :online)))
    g = gt[:, cols]
    mosaic(joinpath(OUT, "$name.mp4"), X, rows, g, dims)
    mosaic(joinpath(OUT, "$name.png"), X, rows, g, dims; frames=busiest - lo + 1:busiest - lo + 1)
    println("  wrote outputs/$name.mp4 (ROI frames $(first(cols))–$(last(cols))) and $name.png")
end

# Randomized (several widths) vs exact Golub–Reinsch SVT on the first window of real frames.
function bench(video="highway")
    BLAS.set_num_threads(1)
    bytes, labels, dims = read_video(joinpath(DATA, video))
    X, gt = bytes[:, 1:WINDOW] ./ 255, labels[:, 1:WINDOW]
    best_f1(S) = maximum(pr_curve(box_filter!(similar(S), abs.(S), dims, 1), gt, TAUS).f1)
    report = IOBuffer()
    line(s) = (println(s); println(report, s))
    line(@sprintf("%s, %d×%d, BLAS threads = 1, randomized: median of 3 warm runs, exact: 1 warm run", video, size(X)...))
    line(@sprintf("%-16s %8s %6s %5s %10s %12s %7s %14s", "SVT", "time", "iters", "rank", "alloc", "alloc/iter", "F1", "‖L-L_exact‖/‖L‖"))
    reference = Matrix{Float64}(undef, 0, 0)   # exact L, computed first
    for (name, width, exact) in (("exact GR", 40, true), ("randomized w=20", 20, false), ("randomized w=40", 40, false), ("randomized w=80", 80, false))
        ws = RPCAWorkspace(X; width)
        rpca!(ws, X; exact, max_iter=2)
        runs = [@timed(rpca!(ws, X; exact, tol=TOL)) for _ in 1:(exact ? 1 : 3)]
        iterations = runs[1].value
        exact && (reference = copy(ws.L))
        allocated = runs[1].bytes
        line(@sprintf("%-16s %7.2fs %6d %5d %8.1fMB %10.2fMB %7.3f %14.1e", name, median(r.time for r in runs), iterations,
                      ws.rank[iterations], allocated / 2^20, allocated / iterations / 2^20, best_f1(ws.S), relative(ws.L, reference)))
    end

    # Memory must stay flat across repeated solves: no growth in live heap.
    ws = RPCAWorkspace(X; width=40)
    rpca!(ws, X; tol=TOL)
    live = map(1:5) do _
        rpca!(ws, X; tol=TOL)
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
    isfile(counts_path(CALIBRATION)) && render(ARGS[2])
elseif command == "render"
    render(ARGS[2]; center=length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing)
elseif command == "summary"
    summary()
elseif command == "sweep"
    online_sweep()
elseif command == "bench"
    bench(get(ARGS, 2, "highway"))
else
    error("usage: run.jl synthetic | cdnet <video> | render <video> [frame] | summary | sweep | bench [video]")
end
