using LinearAlgebra
using Random
using RobustPCA
using RobustPCA: shrink_singular!
using SVDScratch: golub_reinsch_svd, randomized_range!
using Test

function svt_reference(T, τ)
    F = svd(T)
    F.U * Diagonal(max.(F.S .- τ, 0.0)) * F.Vt
end

relative(A, B) = norm(A - B) / norm(B)

@testset "soft threshold" begin
    @test soft(3.0, 1.0) == 2.0
    @test soft(-3.0, 1.0) == -2.0
    @test soft(0.5, 1.0) == 0.0
    @test soft(-0.5, 1.0) == 0.0
    # proximal map of τ|x|: minimizes ½(z - x)² + τ|z|
    for x in -2:0.25:2
        z = soft(x, 0.7)
        @test all(0.5(z - x)^2 + 0.7abs(z) <= 0.5(w - x)^2 + 0.7abs(w) + 1e-12 for w in -3:0.01:3)
    end
end

@testset "singular value thresholding" begin
    rng = Xoshiro(2)
    m, n = 500, 120
    T = randn(rng, m, 6) * Diagonal([40, 30, 20, 10, 5, 3]) * randn(rng, 6, n) ./ sqrt(m) + 1e-3randn(rng, m, n)
    ws = RPCAWorkspace(T; width=20)
    for τ in (0.5, 2.0, 25.0)
        reference = svt_reference(T, τ)
        @test svt_exact!(ws, T, τ) == count(>(τ), svdvals(T))
        @test relative(ws.L, reference) < 1e-12
        @test svt!(ws, T, τ) == count(>(τ), svdvals(T))
        @test relative(ws.L, reference) < 1e-8
    end
    @test svt!(ws, T, 1e3) == 0
    @test iszero(ws.L)
end

@testset "principal component pursuit recovers L0 and S0" begin
    L0, S0 = synthetic(400, 200, 5, 0.05)
    X = L0 + S0
    ws = RPCAWorkspace(X; width=30)
    for exact in (false, true)
        iterations = rpca!(ws, X; exact)
        @test iterations < 60
        @test ws.residual[iterations] < 1e-7
        @test relative(ws.L, L0) < 1e-5
        @test relative(ws.S, S0) < 1e-5
        @test ws.rank[iterations] == 5
    end
end

@testset "no hidden allocations" begin
    rng = Xoshiro(3)
    L0, S0 = synthetic(2000, 100, 4, 0.05; rng)
    X = L0 + S0
    ws = RPCAWorkspace(X; width=20, max_iter=60)
    (; L, S, Y, T) = ws
    μ, λ = 2.0, 0.02
    update!(X, L, S, Y, T, μ, λ) = (S .= soft.(X .- L .+ Y ./ μ, λ / μ); T .= X .- L .- S; Y .+= μ .* T; norm(T))
    update!(X, L, S, Y, T, μ, λ)
    @test (@allocated update!(X, L, S, Y, T, μ, λ)) == 0
    randomized_range!(ws.range, X, 1, ws.rng)
    @test (@allocated randomized_range!(ws.range, X, 1, ws.rng)) == 0
    background, buffer = zeros(size(X, 1)), zeros(size(X, 2))
    median_background!(background, X, buffer)
    @test (@allocated median_background!(background, X, buffer)) == 0

    # Everything per iteration except the small w×n core SVD is allocation-free.
    svt!(ws, X, 0.1)
    core = @allocated golub_reinsch_svd(ws.range.B)
    @test (@allocated svt!(ws, X, 0.1)) <= core
    rpca!(ws, X)
    iterations = rpca!(ws, X)
    @test (@allocated rpca!(ws, X)) <= (iterations + 1) * core + 4096
end

@testset "precision / recall against CDnet labels" begin
    score = [0.9 0.1 0.8 0.7 0.95 0.2]
    gt = UInt8[255 255 0 50 85 170]   # TP, FN, FP, shadow-as-negative, outside ROI, unknown
    curve = pr_curve(score, gt, [0.5, 0.75, 1.0])
    @test curve.recall == [0.5, 0.5, 0.0]
    @test curve.precision ≈ [1 / 3, 0.5, 1.0]
    @test curve.f1 ≈ [0.4, 0.5, 0.0]
    @test_throws ArgumentError pr_curve(score, gt, [0.5, 0.1])
end

@testset "temporal median" begin
    X = [1.0 5 3; 2 2 9; 7 0 1]
    @test median_background!(zeros(3), X) == [3.0, 2.0, 1.0]
end

@testset "selection" begin
    rng = Xoshiro(4)
    for n in (1, 2, 7, 100, 101), _ in 1:20
        a = rand(rng, 0:5, n) .+ 0.0   # many ties
        k = rand(rng, 1:n)
        @test RobustPCA.select!(copy(a), k) == sort(a)[k]
    end
end

@testset "box filter" begin
    A = reshape(Float64.(1:12), 12, 1)            # one 4×3 frame, row-major
    out = box_filter!(similar(A), A, (4, 3), 1)
    frame = reshape(1.0:12, 4, 3)                  # frame[x, y]
    ref = [sum(frame[clamp(x + dx, 1, 4), clamp(y + dy, 1, 3)] for dx in -1:1, dy in -1:1) / 9 for x in 1:4, y in 1:3]
    @test out ≈ vec(ref)
    @test box_filter!(similar(A), A, (4, 3), 0) == A
    buffer = zeros(12)
    box_filter!(out, A, (4, 3), 1, buffer)
    @test (@allocated box_filter!(out, A, (4, 3), 1, buffer)) == 0
end

@testset "counts add across windows" begin
    rng = Xoshiro(5)
    score = rand(rng, 50, 40)
    gt = rand(rng, UInt8[0x00, 0x32, 0x55, 0xaa, 0xff], 50, 40)
    grid = 0.0:0.05:1.0
    whole = tally!(PRCounts(grid), score, gt)
    split = tally!(tally!(PRCounts(grid), view(score, :, 1:15), view(gt, :, 1:15)), view(score, :, 16:40), view(gt, :, 16:40))
    @test whole.positives == split.positives && whole.negatives == split.negatives
    @test pr_curve(tally!(PRCounts(grid ./ 2), score, gt; scale=0.5)).f1 == pr_curve(whole).f1
    counts = PRCounts(grid)
    tally!(counts, score, gt)
    @test (@allocated tally!(counts, score, gt; scale=2.0)) == 0
end


@testset "online RPCA" begin
    rng = Xoshiro(7)
    m, n, r = 3000, 600, 3
    basis = randn(rng, m, r) ./ sqrt(m)
    coefficients = 5 .+ randn(rng, r, n)            # background drifts inside a fixed subspace
    L0 = basis * coefficients
    S0 = zeros(m, n)
    for j in 1:n, i in randperm(rng, m)[1:150]      # 5 % outliers per frame
        S0[i, j] = rand(rng, (-1.0, 1.0))
    end
    X = L0 + S0
    ws = RPCAWorkspace(X[:, 1:200]; width=10)
    rpca!(ws, X[:, 1:200])
    o = OnlineRPCA(ws, r)
    S, L = zeros(m, n), zeros(m, n)
    for j in 1:n
        step!(o, view(S, :, j), view(X, :, j))
        L[:, j] .= o.background
    end
    # Per-frame lasso bias: x − s keeps ±λ₂ at each outlier, so the error is ∝ λ₂ (1.9e-3 here,
    # 9.9e-4 at λ₂/2) rather than the 1e-7 of batch IALM, whose shrinkage vanishes as μ → ∞.
    @test relative(L[:, 201:end], L0[:, 201:end]) < 3e-3
    support = .!iszero.(S0[:, 201:end])
    detected = abs.(S[:, 201:end]) .> 0.5
    @test detected == support
    s = zeros(m)
    step!(o, s, view(X, :, 1))
    @test (@allocated step!(o, s, view(X, :, 1))) == 0

    # An object that stops: batch RPCA over a window where it never moves puts it in L;
    # the online model only learns from x − s, so it stays foreground.
    box = zeros(m); box[1:200] .= 0.8
    Xbox = X[:, 201:400] .+ box
    batch = RPCAWorkspace(Xbox; width=10)
    rpca!(batch, Xbox)
    @test sum(abs, view(batch.S, 1:200, :)) / sum(abs, box[1:200] .* ones(1, 200)) < 0.2
    for j in 1:200
        step!(o, s, view(Xbox, :, j))
    end
    @test sum(abs, view(s, 1:200)) / sum(abs, box) > 0.9
end
