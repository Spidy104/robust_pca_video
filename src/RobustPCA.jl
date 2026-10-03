module RobustPCA

using LinearAlgebra
using Random
using SVDScratch: RandomizedWorkspace, golub_reinsch_svd, randomized_range!

export RPCAWorkspace, rpca!, svt!, svt_exact!, OnlineRPCA, step!, soft, synthetic, median_background!, box_filter!,
       PRCounts, tally!, pr_curve, read_video

"""
Buffers for principal component pursuit on an `m × n` data matrix, allocated once.
`range` holds the randomized range finder (`Q`: m×w, `B = QᵀT`: w×n) and `C` the
thresholded w×n core, so `L = Q*C` without ever forming the m×w singular vectors.
"""
struct RPCAWorkspace
    L::Matrix{Float64}
    S::Matrix{Float64}
    Y::Matrix{Float64}
    T::Matrix{Float64}
    range::RandomizedWorkspace{Float64}
    C::Matrix{Float64}
    rng::Xoshiro
    rank::Vector{Int}
    residual::Vector{Float64}
end

function RPCAWorkspace(m::Integer, n::Integer; width::Integer=40, max_iter::Integer=100)
    1 <= width <= min(m, n) || throw(ArgumentError("width must be in 1:min(m, n)"))
    matrix() = zeros(m, n)
    range = RandomizedWorkspace(Matrix{Float64}(undef, m, width), Matrix{Float64}(undef, n, width), Matrix{Float64}(undef, width, n))
    RPCAWorkspace(matrix(), matrix(), matrix(), matrix(), range, zeros(width, n), Xoshiro(), zeros(Int, max_iter), zeros(max_iter))
end

RPCAWorkspace(X::AbstractMatrix; kwargs...) = RPCAWorkspace(size(X)...; kwargs...)

"Soft-thresholding operator, the proximal map of `τ|x|`."
soft(x, τ) = sign(x) * max(abs(x) - τ, zero(x))

# Given a factorization A = U Σ Vᵀ, write U max(Σ - τ, 0) Vᵀ into dst. Scales U in place.
function shrink_singular!(dst, U, σ, V, τ)
    k = count(>(τ), σ)
    k == 0 && return (fill!(dst, 0.0); 0)
    Uk = view(U, :, 1:k)
    Uk .*= transpose(view(σ, 1:k)) .- τ
    mul!(dst, Uk, adjoint(view(V, :, 1:k)))
    k
end

"""
Singular value thresholding `L = D_τ(T)` from a rank-`width` randomized approximation
`T ≈ Q B` (Halko–Martinsson–Tropp). Only the small `B` is decomposed exactly.
Returns the number of singular values kept; reaching `width` means the cap is binding.
"""
function svt!(ws::RPCAWorkspace, T::AbstractMatrix{Float64}, τ::Real; power_iterations::Integer=1)
    randomized_range!(ws.range, T, power_iterations, ws.rng)
    F = golub_reinsch_svd(ws.range.B)
    k = shrink_singular!(ws.C, F.U, F.S, F.V, τ)
    mul!(ws.L, ws.range.Q, ws.C)
    k
end

"Exact singular value thresholding with the full Golub–Reinsch SVD of `T`. Reference path."
function svt_exact!(ws::RPCAWorkspace, T::AbstractMatrix{Float64}, τ::Real)
    F = golub_reinsch_svd(T)
    shrink_singular!(ws.L, F.U, F.S, F.V, τ)
end

function spectral_norm!(ws::RPCAWorkspace, X)
    randomized_range!(ws.range, X, 2, ws.rng)
    first(golub_reinsch_svd(ws.range.B).S)
end

"""
    rpca!(ws, X; λ, tol, max_iter, ρ, exact, seed) -> iterations

Principal component pursuit `min ‖L‖_* + λ‖S‖₁ s.t. L + S = X` by the inexact
augmented Lagrange multiplier method (Lin, Chen & Ma 2010, Algorithm 5).
Results are in `ws.L` and `ws.S`; per-iteration rank and relative residual in
`ws.rank` and `ws.residual`.
"""
function rpca!(ws::RPCAWorkspace, X::AbstractMatrix{Float64};
               λ::Real=1 / sqrt(maximum(size(X))), tol::Real=1e-7,
               max_iter::Integer=length(ws.rank), ρ::Real=1.5, exact::Bool=false, seed::Integer=0)
    size(X) == size(ws.L) || throw(DimensionMismatch("workspace does not match X"))
    max_iter <= length(ws.rank) || throw(ArgumentError("max_iter exceeds workspace history"))
    (; L, S, Y, T) = ws
    normF = norm(X)
    if iszero(normF)                        # X = 0 is solved by L = S = 0; μ = 1.25/‖X‖₂ would be Inf
        fill!(L, 0.0)
        fill!(S, 0.0)
        return 0
    end
    Random.seed!(ws.rng, seed)
    norm2 = spectral_norm!(ws, X)
    μ = 1.25 / norm2
    μ_max = 1e7 * μ
    Y .= X ./ max(norm2, maximum(abs, X) / λ)
    fill!(S, 0.0)
    iterations = 0
    for k in 1:max_iter
        T .= X .- S .+ Y ./ μ
        ws.rank[k] = exact ? svt_exact!(ws, T, 1 / μ) : svt!(ws, T, 1 / μ)
        S .= soft.(X .- L .+ Y ./ μ, λ / μ)
        T .= X .- L .- S
        Y .+= μ .* T
        μ = min(ρ * μ, μ_max)
        ws.residual[k] = norm(T) / normF
        iterations = k
        ws.residual[k] < tol && break
    end
    iterations
end

"""
Online robust PCA by stochastic optimization (Feng, Xu & Yan 2013). The nuclear norm is
replaced by its factored form ‖L‖_* = min ½(‖U‖²_F + ‖V‖²_F), so each frame `x` costs one
small problem

    min_{v,s} ½‖x − Uv − s‖² + λ₁/2 ‖v‖² + λ₂‖s‖₁

and the background basis `U` (m×r) is refit from the running sums `A = Σ γᵏ v vᵀ` and
`B = Σ γᵏ (x − s) vᵀ`. The forgetting factor γ sets the memory to about 1/(1 − γ) frames.
"""
struct OnlineRPCA
    U::Matrix{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    G::Matrix{Float64}        # UᵀU + λ₁I, Cholesky-factored in place each frame
    v::Vector{Float64}
    background::Vector{Float64}
    λ₁::Float64
    λ₂::Float64
    γ::Float64
end

"""
Start from a converged batch solve in `ws`: with `L ≈ Uₗ Σ Wᵀ` (top `r` triplets),
`U = Uₗ Σ^½` and per-frame coefficients `Σ^½ Wᵀ`, which makes `A = Σ` and `B = U Σ`.
"""
function OnlineRPCA(ws::RPCAWorkspace, r::Integer; λ₁::Real=1 / sqrt(size(ws.L, 1)), λ₂::Real=λ₁, γ::Real=0.999)
    m = size(ws.L, 1)
    1 <= r <= size(ws.range.Q, 2) || throw(ArgumentError("rank must be in 1:width"))
    0 < γ <= 1 || throw(ArgumentError("γ must be in (0, 1]"))
    randomized_range!(ws.range, ws.L, 1, ws.rng)
    F = golub_reinsch_svd(ws.range.B)
    σ = F.S[1:r]
    U = (ws.range.Q * view(F.U, :, 1:r)) .* sqrt.(transpose(σ))
    OnlineRPCA(U, Matrix(Diagonal(σ)), U .* transpose(σ), zeros(r, r), zeros(r), zeros(m), λ₁, λ₂, γ)
end

"""
Process one frame: write its sparse part into `s` (and its background into
`o.background`), then update the basis. Allocation-free.
"""
function step!(o::OnlineRPCA, s::AbstractVector{Float64}, x::AbstractVector{Float64}; max_inner::Integer=50, tol::Real=1e-6)
    (; U, A, B, G, v, background, λ₁, λ₂, γ) = o
    r = length(v)
    # v-subproblem matrix, factored once: U is fixed while this frame is solved.
    mul!(G, transpose(U), U)
    for j in 1:r
        G[j, j] += λ₁
    end
    LAPACK.potrf!('U', G)
    normx2 = max(sum(abs2, x), eps())
    fill!(s, 0.0)
    for _ in 1:max_inner
        background .= x .- s
        mul!(v, transpose(U), background)
        LAPACK.potrs!('U', G, v)                     # v = (UᵀU + λ₁I)⁻¹ Uᵀ(x − s)
        mul!(background, U, v)
        change = 0.0
        @inbounds for i in eachindex(x)
            new = soft(x[i] - background[i], λ₂)     # s = shrink(x − Uv, λ₂)
            change += abs2(new - s[i])
            s[i] = new
        end
        change < tol^2 * normx2 && break
    end
    # Discounted sufficient statistics, then one sweep of block coordinate descent on U.
    @inbounds for k in 1:r, j in 1:r
        A[j, k] = γ * A[j, k] + v[j] * v[k]
    end
    @inbounds for k in 1:r, i in axes(B, 1)
        B[i, k] = γ * B[i, k] + (x[i] - s[i]) * v[k]
    end
    # Exact minimizer of ½tr(Uᵀ(A + λ₁I)U) − tr(UᵀB) over column j, the others fixed:
    # uⱼ ← uⱼ + (bⱼ − U ãⱼ)/Ãⱼⱼ with Ã = A + λ₁I (Feng, Xu & Yan 2013, Algorithm 2).
    @inbounds for j in 1:r
        scale = 1 / (A[j, j] + λ₁)
        for i in axes(U, 1)
            Ua = λ₁ * U[i, j]
            for k in 1:r
                Ua += U[i, k] * A[k, j]
            end
            U[i, j] += (B[i, j] - Ua) * scale
        end
    end
    s
end

"""
Random test problem from Candès et al. (2011): `L0 = A Bᵀ` with Gaussian factors of
rank `r`, `S0` with `density·m·n` entries of ±1 at uniformly random positions.
"""
function synthetic(m::Integer, n::Integer, r::Integer, density::Real; rng::AbstractRNG=Xoshiro(1))
    L0 = randn(rng, m, r) * randn(rng, r, n) ./ sqrt(m)
    S0 = zeros(m, n)
    for i in randperm(rng, m * n)[1:round(Int, density * m * n)]
        S0[i] = rand(rng, (-1.0, 1.0))
    end
    L0, S0
end

"k-th smallest element of `a`, in place (Wirth's selection, Hoare partitioning)."
function select!(a::AbstractVector, k::Integer)
    lo, hi = firstindex(a), lastindex(a)
    @inbounds while lo < hi
        pivot = a[k]
        i, j = lo, hi
        while i <= j
            while a[i] < pivot; i += 1; end
            while pivot < a[j]; j -= 1; end
            if i <= j
                a[i], a[j] = a[j], a[i]
                i += 1; j -= 1
            end
        end
        j < k && (lo = i)
        k < i && (hi = j)
    end
    a[k]
end

"Per-pixel temporal median of the frames (columns of X), a classic background baseline."
function median_background!(background::AbstractVector{Float64}, X::AbstractMatrix{Float64}, buffer::Vector{Float64}=Vector{Float64}(undef, size(X, 2)))
    middle = cld(size(X, 2), 2)
    @inbounds for i in axes(X, 1)
        copyto!(buffer, view(X, i, :))
        background[i] = select!(buffer, middle)
    end
    background
end

"""
Separable moving average of radius `r` over each frame (column of `A`, `w × h` pixels,
row-major), with clamped edges. Smoothing the foreground score before thresholding fills
the hollow interiors of flat objects; `buffer` holds one frame.
"""
function box_filter!(out::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, dims::NTuple{2,Int}, r::Integer,
                     buffer::Vector{Float64}=Vector{Float64}(undef, prod(dims)))
    w, h = dims
    scale = 1 / (2r + 1)
    @inbounds for j in axes(A, 2)
        for y in 1:h, x in 1:w
            total = 0.0
            for d in -r:r
                total += A[(y - 1) * w + clamp(x + d, 1, w), j]
            end
            buffer[(y - 1) * w + x] = scale * total
        end
        for y in 1:h, x in 1:w
            total = 0.0
            for d in -r:r
                total += buffer[(clamp(y + d, 1, h) - 1) * w + x]
            end
            out[(y - 1) * w + x, j] = scale * total
        end
    end
    out
end

"""
Confusion histograms of the detector `score > t` over a sorted threshold grid, against CDnet
ground truth: 255 = motion is positive, 0 = static and 50 = shadow are negative, 85 = outside
ROI and 170 = unknown are ignored. Bin `k` counts pixels whose score exceeds
`thresholds[1:k-1]`, so counts from separate windows simply add.
"""
struct PRCounts
    thresholds::Vector{Float64}
    positives::Vector{Int}
    negatives::Vector{Int}
end

function PRCounts(thresholds::AbstractVector{<:Real})
    issorted(thresholds) || throw(ArgumentError("thresholds must be sorted"))
    K = length(thresholds)
    PRCounts(collect(Float64, thresholds), zeros(Int, K + 1), zeros(Int, K + 1))
end

"Add the pixels of `scale · score` to the histograms; one pass, one binary search per pixel."
function tally!(counts::PRCounts, score::AbstractArray{<:Real}, gt::AbstractArray{UInt8}; scale::Real=1.0)
    size(score) == size(gt) || throw(DimensionMismatch("score and ground truth differ in size"))
    (; thresholds, positives, negatives) = counts
    @inbounds for i in eachindex(score, gt)
        label = gt[i]
        (label == 0x55 || label == 0xaa) && continue
        bin = searchsortedfirst(thresholds, scale * score[i])
        label == 0xff ? (positives[bin] += 1) : (negatives[bin] += 1)
    end
    counts
end

"Precision, recall and F1 at every threshold of the grid."
function pr_curve(counts::PRCounts)
    total = sum(counts.positives)
    tp = reverse!(cumsum(reverse(counts.positives)))[2:end]   # tp[k]: motion pixels scoring above thresholds[k]
    fp = reverse!(cumsum(reverse(counts.negatives)))[2:end]
    precision = @. ifelse(tp + fp == 0, 1.0, tp / (tp + fp))
    recall = tp ./ max(total, 1)
    f1 = @. ifelse(precision + recall == 0, 0.0, 2precision * recall / (precision + recall))
    (; precision, recall, f1)
end

pr_curve(score::AbstractArray{<:Real}, gt::AbstractArray{UInt8}, thresholds::AbstractVector{<:Real}) =
    pr_curve(tally!(PRCounts(thresholds), score, gt))

"""
Load `data/<video>` written by `scripts/fetch_cdnet.sh`: 8-bit luma frames as columns
(kept as bytes so long videos fit; convert one window at a time), ground-truth labels in
the same layout, and the frame `(width, height)`.
"""
function read_video(dir::AbstractString)
    width, height, frames = parse.(Int, split(read(joinpath(dir, "size.txt"), String)))
    bytes = read!(joinpath(dir, "frames.u8"), Matrix{UInt8}(undef, width * height, frames))
    gt = read!(joinpath(dir, "gt.u8"), Matrix{UInt8}(undef, width * height, frames))
    bytes, gt, (width, height)
end

end
