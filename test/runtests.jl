using LinearAlgebra
using Random
using RobustPCA
using RobustPCA: Step, update!, select!, solve_frame!, factor_gram!
using SVDScratch: golub_reinsch_svd, randomized_range!
using Test

relative(A, B) = norm(A - B) / norm(B)

function svt_reference(T, τ)
    F = svd(T)
    F.U * Diagonal(max.(F.S .- τ, 0.0)) * F.Vt
end

function smooth_reference(residual, (w, h))
    frame = reshape(abs.(residual), w, h)
    vec([sum(frame[clamp(x + dx, 1, w), clamp(y + dy, 1, h)] for dx in -1:1, dy in -1:1) / 9 for x in 1:w, y in 1:h])
end

function lasso_reference(U, x, λ₁, λ₂)
    gram = factorize(Symmetric(U' * U + λ₁ * I))
    s = zeros(length(x))
    for _ in 1:20000
        s_next = soft.(x .- U * (gram \ (U' * (x - s))), λ₂)
        norm(s_next - s) < 1e-13 && return s_next
        s = s_next
    end
    s
end

@testset "soft threshold" begin
    @test soft(3.0, 1.0) == 2.0
    @test soft(-3.0, 1.0) == -2.0
    @test soft(0.5, 1.0) == 0.0
    @test soft(-0.5, 1.0) == 0.0
    for x in -2:0.25:2
        z = soft(x, 0.7)
        @test all(0.5(z - x)^2 + 0.7abs(z) <= 0.5(w - x)^2 + 0.7abs(w) + 1e-12 for w in -3:0.01:3)
    end
end

@testset "singular value thresholding" begin
    rng = Xoshiro(2)
    m, n = 500, 120
    T = randn(rng, m, 6) * Diagonal([40, 30, 20, 10, 5, 3]) * randn(rng, 6, n) ./ sqrt(m) + 1e-3randn(rng, m, n)
    ws = RPCAWorkspace(T, 20)
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
    L0, S0 = synthetic(Xoshiro(1), (400, 200), 5, 0.05)
    X = L0 + S0
    ws = RPCAWorkspace(X, 30)
    for exact in (false, true)
        result = rpca!(ws, X, PCPOptions(exact=exact))
        @test result.iterations < 60
        @test result.residual < 1e-7
        @test result.rank == 5
        @test relative(ws.L, L0) < 1e-5
        @test relative(ws.S, S0) < 1e-5
    end
end

@testset "solves are deterministic" begin
    L0, S0 = synthetic(Xoshiro(9), (300, 80), 4, 0.05)
    X = L0 + S0
    ws = RPCAWorkspace(X, 12)
    rpca!(ws, X)
    first_L, first_S = copy(ws.L), copy(ws.S)
    rpca!(ws, X)
    @test ws.L == first_L && ws.S == first_S
end

@testset "fused update equals the five separate passes" begin
    rng = Xoshiro(10)
    X, L, S, Y = (randn(rng, 40, 30) for _ in 1:4)
    ws = RPCAWorkspace(X, 5)
    ws.L .= L; ws.S .= S; ws.Y .= Y
    μ, μ_next, τ = 2.5, 3.75, 0.3
    residual = update!(ws, X, Step(μ, μ_next, τ))
    S_ref = soft.(X .- L .+ Y ./ μ, τ)
    Z_ref = X .- L .- S_ref
    Y_ref = Y .+ μ .* Z_ref
    @test ws.S == S_ref
    @test ws.Y == Y_ref
    @test ws.T == X .- S_ref .+ Y_ref ./ μ_next
    @test residual ≈ norm(Z_ref)
end

@testset "zero input" begin
    ws = RPCAWorkspace(30, 10, 4)
    @test rpca!(ws, zeros(30, 10)).iterations == 0
    @test iszero(ws.L) && iszero(ws.S)
    online = OnlineRPCA(ws, OnlineOptions(rank=2))
    s = ones(30)
    step!(online, s, zeros(30))
    @test iszero(s) && all(isfinite, online.U)
end

@testset "no hidden allocations" begin
    L0, S0 = synthetic(Xoshiro(3), (2000, 100), 4, 0.05)
    X = L0 + S0
    ws = RPCAWorkspace(X, 20)
    step = Step(2.0, 3.0, 0.01)
    update!(ws, X, step)
    @test (@allocated update!(ws, X, step)) == 0
    randomized_range!(ws.sketch, X, 1, ws.rng)
    @test (@allocated randomized_range!(ws.sketch, X, 1, ws.rng)) == 0
    background, buffer = zeros(size(X, 1)), zeros(size(X, 2))
    median_background!(background, X, buffer)
    @test (@allocated median_background!(background, X, buffer)) == 0
    svt!(ws, X, 0.1)
    core = @allocated golub_reinsch_svd(ws.sketch.B)
    @test (@allocated svt!(ws, X, 0.1)) <= core
    result = rpca!(ws, X)
    @test (@allocated rpca!(ws, X)) <= (result.iterations + 1) * core + 4096
end

@testset "smoother matches the 3×3 box average with clamped edges" begin
    rng = Xoshiro(11)
    for dims in ((4, 3), (7, 5), (2, 6), (9, 1))
        sm = Smoother(dims)
        frame = randn(rng, prod(dims))
        @test smooth!(sm, frame) ≈ smooth_reference(frame, dims)
    end
    sm = Smoother((16, 12))
    frame = randn(rng, 16 * 12)
    smooth!(sm, frame)
    @test (@allocated smooth!(sm, frame)) == 0
    @test_throws ArgumentError Smoother((1, 5))
    @test_throws DimensionMismatch smooth!(sm, zeros(5))
end

@testset "precision, recall and F1 against CDnet labels" begin
    counts = PRCounts(0.25, 3)
    score = [0.9, 0.1, 0.8, 0.7, 0.95, 0.2]
    gt = UInt8[255, 255, 0, 50, 85, 170]
    tally!(counts, score, gt)
    @test thresholds(counts) ≈ [0.25, 0.5, 0.75]
    curve = pr_curve(counts)
    @test curve.recall == [0.5, 0.5, 0.5]
    @test curve.precision ≈ [1 / 3, 1 / 3, 1 / 2]
    @test curve.f1 ≈ [0.4, 0.4, 0.5]
    @test_throws DimensionMismatch tally!(counts, score, gt[1:3])
end

@testset "counts add across windows and bin like a binary search" begin
    rng = Xoshiro(5)
    score = rand(rng, 2000)
    gt = rand(rng, UInt8[0x00, 0x32, 0x55, 0xaa, 0xff], 2000)
    whole = tally!(PRCounts(0.05, 20), score, gt)
    halves = tally!(tally!(PRCounts(0.05, 20), score[1:700], gt[1:700]), score[701:end], gt[701:end])
    @test whole.positives == halves.positives && whole.negatives == halves.negatives
    grid = thresholds(whole)
    keep = gt .∉ Ref((0x55, 0xaa))
    expected = [count(i -> keep[i] && gt[i] == 0xff && score[i] > t, 1:2000) for t in grid]
    @test reverse(cumsum(reverse(whole.positives)))[2:end] == expected
    @test (@allocated tally!(whole, score, gt)) == 0
end

@testset "temporal median and selection" begin
    X = [1.0 5 3; 2 2 9; 7 0 1]
    @test median_background!(zeros(3), X, zeros(3)) == [3.0, 2.0, 1.0]
    rng = Xoshiro(4)
    for n in (1, 2, 7, 100, 101), _ in 1:20
        a = rand(rng, 0:5, n) .+ 0.0
        k = rand(rng, 1:n)
        @test select!(copy(a), k) == sort(a)[k]
    end
end

@testset "reading a video" begin
    mktempdir() do dir
        write(joinpath(dir, "size.txt"), "4 3 2\n")
        write(joinpath(dir, "frames.u8"), UInt8.(1:24))
        write(joinpath(dir, "gt.u8"), UInt8.(24:-1:1))
        bytes, gt, dims = read_video(dir)
        @test dims == (4, 3)
        @test size(bytes) == (12, 2) && bytes[:, 2] == UInt8.(13:24)
        @test gt[1, 1] == 24
    end
end

@testset "online frame solve reaches the lasso optimum" begin
    rng = Xoshiro(12)
    m, r = 400, 4
    ws = RPCAWorkspace(m, 30, 6)
    online = OnlineRPCA(ws, OnlineOptions(rank=4, tol=1e-9, max_inner=5000))
    online.U .= randn(rng, m, r) ./ sqrt(m)
    x = online.U * randn(rng, r) .+ [i <= 20 ? 1.0 : 0.0 for i in 1:m] .+ 0.01randn(rng, m)
    s = zeros(m)
    factor_gram!(online)
    iterations = solve_frame!(online, s, x, sum(abs2, x))
    @test iterations < 5000
    @test relative(s, lasso_reference(online.U, x, online.λ₁, online.λ₂)) < 1e-6
end

@testset "online RPCA tracks a drifting background" begin
    rng = Xoshiro(7)
    m, n, r = 3000, 600, 3
    basis = randn(rng, m, r) ./ sqrt(m)
    L0 = basis * (5 .+ randn(rng, r, n))
    S0 = zeros(m, n)
    for j in 1:n, i in randperm(rng, m)[1:150]
        S0[i, j] = rand(rng, (-1.0, 1.0))
    end
    X = L0 + S0
    ws = RPCAWorkspace(X[:, 1:200], 10)
    rpca!(ws, X[:, 1:200])
    online = OnlineRPCA(ws, OnlineOptions(rank=r))
    S, L = zeros(m, n), zeros(m, n)
    for j in 1:n
        step!(online, view(S, :, j), view(X, :, j))
        L[:, j] .= online.background
    end
    @test relative(L[:, 201:end], L0[:, 201:end]) < 3e-3
    @test (abs.(S[:, 201:end]) .> 0.5) == .!iszero.(S0[:, 201:end])
    s = zeros(m)
    step!(online, s, view(X, :, 1))
    @test (@allocated step!(online, s, view(X, :, 1))) == 0
end

@testset "an object that stops stays foreground online but not in batch" begin
    rng = Xoshiro(14)
    m, n, r = 3000, 400, 3
    L0 = (randn(rng, m, r) ./ sqrt(m)) * (5 .+ randn(rng, r, n))
    X = L0 .+ 0.001randn(rng, m, n)
    box = zeros(m)
    box[1:200] .= 0.8
    ws = RPCAWorkspace(X[:, 1:200], 10)
    rpca!(ws, X[:, 1:200])
    online = OnlineRPCA(ws, OnlineOptions(rank=r))
    s = zeros(m)
    for j in 201:400
        step!(online, s, X[:, j] .+ box)
    end
    @test sum(abs, view(s, 1:200)) / sum(box) > 0.9
    stopped = X[:, 201:400] .+ box
    batch = RPCAWorkspace(stopped, 10)
    rpca!(batch, stopped)
    @test sum(abs, view(batch.S, 1:200, :)) / (200 * sum(box)) < 0.2
end

@testset "online basis update is the exact coordinate minimizer" begin
    online = OnlineRPCA(RPCAWorkspace(1, 2, 1), OnlineOptions(rank=1, γ=1.0))
    online.U .= 1.0
    online.A .= 2.0
    online.B .= 2.0
    step!(online, zeros(1), zeros(1))
    @test online.U[1, 1] ≈ 2 / 3

    rng = Xoshiro(8)
    m, r = 50, 3
    online = OnlineRPCA(RPCAWorkspace(m, 10, r), OnlineOptions(rank=r, γ=1.0))
    F = randn(rng, r, r)
    online.A .= F * F' + I
    online.B .= randn(rng, m, r)
    online.U .= randn(rng, m, r)
    for _ in 1:200
        step!(online, zeros(m), zeros(m))
    end
    @test online.U ≈ online.B / (online.A + online.λ₁ * I)
end

@testset "non-finite input is rejected before any state changes" begin
    L0, S0 = synthetic(Xoshiro(15), (200, 40), 3, 0.05)
    X = L0 + S0
    ws = RPCAWorkspace(X, 5)
    rpca!(ws, X)
    online = OnlineRPCA(ws, OnlineOptions(rank=3))
    U, A, B = copy(online.U), copy(online.A), copy(online.B)
    s = zeros(200)
    for bad in (NaN, Inf, -Inf)
        frame = X[:, 1]
        frame[7] = bad
        @test_throws ArgumentError step!(online, s, frame)
        @test online.U == U && online.A == A && online.B == B
    end
    step!(online, s, X[:, 2])
    @test all(isfinite, s)
    X[3, 4] = NaN
    @test_throws ArgumentError rpca!(ws, X)
end

@testset "invalid arguments are rejected" begin
    ws = RPCAWorkspace(20, 10, 3)
    @test_throws ArgumentError RPCAWorkspace(20, 10, 11)
    @test_throws DimensionMismatch rpca!(ws, zeros(21, 10))
    @test_throws ArgumentError OnlineRPCA(ws, OnlineOptions(rank=4))
    @test_throws ArgumentError OnlineRPCA(ws, OnlineOptions(rank=2, γ=1.5))
end
