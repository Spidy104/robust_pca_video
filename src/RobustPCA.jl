module RobustPCA

using LinearAlgebra
using Random
using SVDScratch: RandomizedWorkspace, golub_reinsch_svd, randomized_range!

export RPCAWorkspace, PCPOptions, rpca!, svt!, svt_exact!
export OnlineRPCA, OnlineOptions, step!
export PRCounts, Smoother, smooth!, tally!, pr_curve, thresholds
export soft, synthetic, median_background!, read_video

soft(x, τ) = copysign(max(abs(x) - τ, zero(x)), x)

struct RPCAWorkspace
    L::Matrix{Float64}
    S::Matrix{Float64}
    Y::Matrix{Float64}
    T::Matrix{Float64}
    sketch::RandomizedWorkspace{Float64}
    core::Matrix{Float64}
    rng::Xoshiro
end

function RPCAWorkspace(m::Integer, n::Integer, width::Integer)
    1 <= width <= min(m, n) || throw(ArgumentError("width must be in 1:min(m, n)"))
    sketch = RandomizedWorkspace(Matrix{Float64}(undef, m, width), Matrix{Float64}(undef, n, width), Matrix{Float64}(undef, width, n))
    RPCAWorkspace(zeros(m, n), zeros(m, n), zeros(m, n), zeros(m, n), sketch, zeros(width, n), Xoshiro(0))
end

RPCAWorkspace(X::AbstractMatrix, width::Integer) = RPCAWorkspace(size(X)..., width)

Base.@kwdef struct PCPOptions
    tol::Float64 = 1e-7
    max_iter::Int = 100
    ρ::Float64 = 1.5
    exact::Bool = false
end

function shrink!(dst::AbstractMatrix{Float64}, F, τ::Real)
    kept = count(>(τ), F.S)
    kept == 0 && return (fill!(dst, 0.0); 0)
    U = view(F.U, :, 1:kept)
    U .*= transpose(view(F.S, 1:kept)) .- τ
    mul!(dst, U, adjoint(view(F.V, :, 1:kept)))
    kept
end

function svt!(ws::RPCAWorkspace, T::AbstractMatrix{Float64}, τ::Real)
    randomized_range!(ws.sketch, T, 1, ws.rng)
    kept = shrink!(ws.core, golub_reinsch_svd(ws.sketch.B), τ)
    mul!(ws.L, ws.sketch.Q, ws.core)
    kept
end

svt_exact!(ws::RPCAWorkspace, T::AbstractMatrix{Float64}, τ::Real) = shrink!(ws.L, golub_reinsch_svd(T), τ)

function spectral_norm!(ws::RPCAWorkspace, X::AbstractMatrix{Float64})
    randomized_range!(ws.sketch, X, 2, ws.rng)
    first(golub_reinsch_svd(ws.sketch.B).S)
end

struct Step
    μ::Float64
    μ_next::Float64
    τ::Float64
end

function update!(ws::RPCAWorkspace, X::AbstractMatrix{Float64}, step::Step)
    (; L, S, Y, T) = ws
    (; μ, μ_next, τ) = step
    squares = 0.0
    @inbounds @simd for i in eachindex(X, L, S, Y, T)
        x, l, y = X[i], L[i], Y[i]
        s = soft(x - l + y / μ, τ)
        z = x - l - s
        y_next = y + μ * z
        S[i] = s
        Y[i] = y_next
        T[i] = x - s + y_next / μ_next
        squares += z * z
    end
    sqrt(squares)
end

function zero_solution!(ws::RPCAWorkspace)
    fill!(ws.L, 0.0)
    fill!(ws.S, 0.0)
    (; iterations=0, rank=0, residual=0.0)
end

function rpca!(ws::RPCAWorkspace, X::AbstractMatrix{Float64}, opts::PCPOptions=PCPOptions())
    size(X) == size(ws.L) || throw(DimensionMismatch("workspace does not match X"))
    norm_X = norm(X)
    isfinite(norm_X) || throw(ArgumentError("X must be finite"))
    iszero(norm_X) && return zero_solution!(ws)
    Random.seed!(ws.rng, 0)
    norm2 = spectral_norm!(ws, X)
    λ = 1 / sqrt(maximum(size(X)))
    μ = 1.25 / norm2
    μ_max = 1e7 * μ
    ws.Y .= X ./ max(norm2, maximum(abs, X) / λ)
    ws.T .= X .+ ws.Y ./ μ
    rank, residual = 0, Inf
    for iteration in 1:opts.max_iter
        rank = opts.exact ? svt_exact!(ws, ws.T, 1 / μ) : svt!(ws, ws.T, 1 / μ)
        μ_next = min(opts.ρ * μ, μ_max)
        residual = update!(ws, X, Step(μ, μ_next, λ / μ)) / norm_X
        μ = μ_next
        residual < opts.tol && return (; iterations=iteration, rank, residual)
    end
    (; iterations=opts.max_iter, rank, residual)
end

Base.@kwdef struct OnlineOptions
    rank::Int = 5
    γ::Float64 = 0.999
    λ₂::Float64 = 1.0
    tol::Float64 = 1e-4
    max_inner::Int = 100
end

struct OnlineRPCA
    U::Matrix{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    gram::Matrix{Float64}
    v::Vector{Float64}
    background::Vector{Float64}
    work::Vector{Float64}
    λ₁::Float64
    λ₂::Float64
    γ::Float64
    tol::Float64
    max_inner::Int
end

function OnlineRPCA(ws::RPCAWorkspace, opts::OnlineOptions)
    m, r = size(ws.L, 1), opts.rank
    1 <= r <= size(ws.sketch.Q, 2) || throw(ArgumentError("rank must be in 1:width"))
    0 < opts.γ <= 1 || throw(ArgumentError("γ must be in (0, 1]"))
    randomized_range!(ws.sketch, ws.L, 1, ws.rng)
    F = golub_reinsch_svd(ws.sketch.B)
    σ = F.S[1:r]
    U = (ws.sketch.Q * view(F.U, :, 1:r)) .* sqrt.(transpose(σ))
    λ₁ = 1 / sqrt(m)
    OnlineRPCA(U, Matrix(Diagonal(σ)), U .* transpose(σ), zeros(r, r), zeros(r), zeros(m), zeros(m),
               λ₁, opts.λ₂ * λ₁, opts.γ, opts.tol, opts.max_inner)
end

function factor_gram!(o::OnlineRPCA)
    (; U, gram, λ₁) = o
    @inbounds for j in axes(U, 2), i in 1:j
        total = 0.0
        @simd for k in axes(U, 1)
            total += U[k, i] * U[k, j]
        end
        gram[i, j] = i == j ? total + λ₁ : total
    end
    LAPACK.potrf!('U', gram)
end

function fit_coefficients!(o::OnlineRPCA, target::AbstractVector{Float64})
    mul!(o.v, transpose(o.U), target)
    LAPACK.potrs!('U', o.gram, o.v)
    mul!(o.background, o.U, o.v)
end

function shrink_extrapolate!(o::OnlineRPCA, s::AbstractVector{Float64}, x::AbstractVector{Float64}, β::Float64)
    (; background, work, λ₂) = o
    change = 0.0
    @inbounds @simd for i in eachindex(s, x, background, work)
        s_new = soft(x[i] - background[i], λ₂)
        d = s_new - s[i]
        change += d * d
        work[i] = s_new + β * d
        s[i] = s_new
    end
    change
end

function solve_frame!(o::OnlineRPCA, s::AbstractVector{Float64}, x::AbstractVector{Float64}, energy::Float64)
    stop = o.tol^2 * max(energy, eps())
    fill!(s, 0.0)
    fill!(o.work, 0.0)
    momentum = 1.0
    for iteration in 1:o.max_inner
        o.background .= x .- o.work
        fit_coefficients!(o, o.background)
        momentum_next = (1 + sqrt(1 + 4momentum^2)) / 2
        change = shrink_extrapolate!(o, s, x, (momentum - 1) / momentum_next)
        momentum = momentum_next
        change < stop && return iteration
    end
    o.max_inner
end

function accumulate!(o::OnlineRPCA)
    (; A, B, v, work, γ) = o
    A .= γ .* A .+ v .* transpose(v)
    for k in eachindex(v)
        @views B[:, k] .= γ .* B[:, k] .+ v[k] .* work
    end
end

function sweep_basis!(o::OnlineRPCA)
    (; U, A, B, work, λ₁) = o
    for j in axes(U, 2)
        mul!(work, U, view(A, :, j))
        scale = 1 / (A[j, j] + λ₁)
        @views U[:, j] .+= (B[:, j] .- work .- λ₁ .* U[:, j]) .* scale
    end
end

function step!(o::OnlineRPCA, s::AbstractVector{Float64}, x::AbstractVector{Float64})
    energy = sum(abs2, x)
    isfinite(energy) || throw(ArgumentError("frame must be finite"))
    factor_gram!(o)
    iterations = solve_frame!(o, s, x, energy)
    o.work .= x .- s
    fit_coefficients!(o, o.work)
    accumulate!(o)
    sweep_basis!(o)
    iterations
end

function synthetic(rng::AbstractRNG, shape::NTuple{2,Int}, r::Integer, density::Real)
    m, n = shape
    L0 = randn(rng, m, r) * randn(rng, r, n) ./ sqrt(m)
    S0 = zeros(m, n)
    for i in randperm(rng, m * n)[1:round(Int, density * m * n)]
        S0[i] = rand(rng, (-1.0, 1.0))
    end
    L0, S0
end

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

function median_background!(background::AbstractVector{Float64}, X::AbstractMatrix{Float64}, buffer::Vector{Float64})
    middle = cld(size(X, 2), 2)
    @inbounds for i in axes(X, 1)
        copyto!(buffer, view(X, i, :))
        background[i] = select!(buffer, middle)
    end
    background
end

struct Smoother
    dims::NTuple{2,Int}
    out::Vector{Float64}
    tmp::Vector{Float64}
end

function Smoother(dims::NTuple{2,Int})
    dims[1] >= 2 && dims[2] >= 1 || throw(ArgumentError("frames must be at least 2 pixels wide"))
    Smoother(dims, zeros(prod(dims)), zeros(prod(dims)))
end

function smooth!(sm::Smoother, residual::AbstractVector{Float64})
    length(residual) == length(sm.out) || throw(DimensionMismatch("residual does not match the frame size"))
    w, h = sm.dims
    (; out, tmp) = sm
    third = 1 / 3
    @inbounds for y in 0:h - 1
        row = y * w
        tmp[row + 1] = (abs(residual[row + 1]) + abs(residual[row + 1]) + abs(residual[row + 2])) * third
        @simd for x in 2:w - 1
            tmp[row + x] = (abs(residual[row + x - 1]) + abs(residual[row + x]) + abs(residual[row + x + 1])) * third
        end
        tmp[row + w] = (abs(residual[row + w - 1]) + abs(residual[row + w]) + abs(residual[row + w])) * third
    end
    @inbounds for y in 1:h
        above, below = (max(y - 1, 1) - 1) * w, (min(y + 1, h) - 1) * w
        row = (y - 1) * w
        @simd for x in 1:w
            out[row + x] = (tmp[above + x] + tmp[row + x] + tmp[below + x]) * third
        end
    end
    out
end

struct PRCounts
    step::Float64
    positives::Vector{Int}
    negatives::Vector{Int}
end

PRCounts(step::Real, bins::Integer) = PRCounts(step, zeros(Int, bins + 1), zeros(Int, bins + 1))

thresholds(counts::PRCounts) = counts.step .* (1:length(counts.positives) - 1)

function tally!(counts::PRCounts, score::AbstractVector{Float64}, gt::AbstractVector{UInt8})
    length(score) == length(gt) || throw(DimensionMismatch("score and ground truth differ in size"))
    (; positives, negatives) = counts
    inverse_step, top = 1 / counts.step, length(positives) - 1
    @inbounds for i in eachindex(score, gt)
        label = gt[i]
        (label == 0x55 || label == 0xaa) && continue
        exceeded = clamp(unsafe_trunc(Int, ceil(score[i] * inverse_step)) - 1, 0, top)
        label == 0xff ? (positives[exceeded + 1] += 1) : (negatives[exceeded + 1] += 1)
    end
    counts
end

function pr_curve(counts::PRCounts)
    tp = reverse!(cumsum(reverse(counts.positives)))[2:end]
    fp = reverse!(cumsum(reverse(counts.negatives)))[2:end]
    precision = @. ifelse(tp + fp == 0, 1.0, tp / (tp + fp))
    recall = tp ./ max(sum(counts.positives), 1)
    f1 = @. ifelse(precision + recall == 0, 0.0, 2precision * recall / (precision + recall))
    (; precision, recall, f1)
end

function read_video(dir::AbstractString)
    width, height, frames = parse.(Int, split(read(joinpath(dir, "size.txt"), String)))
    bytes = read!(joinpath(dir, "frames.u8"), Matrix{UInt8}(undef, width * height, frames))
    gt = read!(joinpath(dir, "gt.u8"), Matrix{UInt8}(undef, width * height, frames))
    bytes, gt, (width, height)
end

end
