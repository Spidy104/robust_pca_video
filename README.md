# Robust PCA video background separation

Splits a fixed-camera video into a low-rank background `L` and a sparse foreground `S` by principal component pursuit,

```
min ‖L‖_* + λ‖S‖₁   subject to   L + S = X,      λ = 1/√max(m, n)
```

with each grayscale frame as one column of `X`. It's solved with the inexact augmented Lagrange multiplier method (Lin, Chen & Ma 2010). Each iteration does singular value thresholding on the background and soft thresholding on the foreground. The SVT step uses the randomized range finder and the Golub–Reinsch SVD from [`svd_from_scratch`](../svd_from_scratch), and only a small w×n core is decomposed exactly.

A second solver, **online RPCA** (OR-PCA, Feng, Xu & Yan 2013), processes one frame at a time. It uses the factored form ‖L‖_* = min ½(‖U‖²_F + ‖V‖²_F), so each frame `x` only needs a small problem

```
min_{v,s} ½‖x − Uv − s‖² + λ₁/2 ‖v‖² + λ₂‖s‖₁        λ₁ = λ₂ = 1/√m
```

solved by alternating a ridge solve `v = (UᵀU + λ₁I)⁻¹Uᵀ(x − s)` and a soft threshold `s = shrink(x − Uv, λ₂)`. The basis `U` (m×5) is then refit by one sweep of block coordinate descent on the discounted sums `A = Σγᵏvvᵀ` and `B = Σγᵏ(x − s)vᵀ`. Its memory is ~1/(1 − γ) frames (γ = 0.999). It is warm-started from the batch solve of the first window.

Everything is scored against CDnet 2014 pixel-level ground truth.

## Run

Julia 1.13, ffmpeg, curl. Standard library plus `SVDScratch` (path dependency) only.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl

scripts/fetch_cdnet.sh baseline highway              # streams one video, 3.5 min at ~3 MB/s
julia --project=scripts scripts/run.jl cdnet highway   # all four methods over the full video; mosaic mp4/png
julia --project=scripts scripts/run.jl summary         # calibrate τ on highway, score every video
julia --project=scripts scripts/run.jl sweep           # online γ and λ₂ sensitivity
julia --project=scripts scripts/run.jl render boats 5500   # batch vs online around a given frame
julia --project=scripts scripts/run.jl bench           # exact vs randomized SVT
julia --project=scripts scripts/run.jl synthetic       # recovery phase grid
```

`fetch_cdnet.sh <category> <video> [offset=0] [frames=all] [width=160]` pulls from the Hugging Face mirror
[`tauseefakhund/D2-CDNet`](https://huggingface.co/datasets/tauseefakhund/D2-CDNet), because the original changedetection.net host no longer resolves. It extracts the video's input frames, ground truth and temporal ROI, and stops downloading once the tar stream moves past the video. ffmpeg then writes the whole temporal ROI (or `frames` of it, from `offset`) as raw 8-bit luma (BT.601, area-averaged to 160 px wide) and the labels with nearest-neighbour scaling, so the label values survive.

## Layout

`src/RobustPCA.jl` holds the whole package: the workspace, `soft`, `svt!`/`svt_exact!`, `rpca!`, `OnlineRPCA`/`step!`, the temporal-median baseline (Wirth's selection), a separable box filter, PR/F1 against CDnet labels, and raw video I/O. `scripts/run.jl` runs the experiments, `test/runtests.jl` the tests, and `outputs/` holds the saved results.

## Memory

Every m×n buffer (`L, S, Y, T`, the range-finder `Q`, the w×n core) is allocated once in `RPCAWorkspace`. The only allocation inside an iteration is the exact SVD of the small w×n core (0.15 MB/iteration at w = 20). Online RPCA allocates once, in `OnlineRPCA`; `step!` allocates nothing per frame. The tests check that the broadcast updates, the range finder, `step!`, the median, the box filter and the PR tally allocate 0 bytes. The live heap stays flat across repeated solves.

## Results

### Synthetic exact recovery (`outputs/synthetic.csv`)

4800×200, `L0` of rank r with Gaussian factors, `S0` of ±1 at random positions, tol 1e-7, w = 60.
Recovery is exact, to the 1e-7 tolerance, below the phase transition and breaks down past it, as the theory predicts:

| rank \ density | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| 5  | 4e-7 | 8e-7 | 4e-7 | 1e-6 | 2e-6 |
| 10 | 3e-7 | 3e-7 | 5e-7 | 9e-7 | 5e-3 |
| 20 | 2e-7 | 3e-7 | 6e-7 | 4e-3 | 4e-2 |
| 30 | 2e-7 | 2e-7 | 2e-3 | 3e-2 | 0.15 |
| 40 | 7e-4 | 3e-3 | 1e-2 | 0.11 | 0.30 |

(relative error ‖L − L0‖/‖L0‖)

### Exact vs randomized SVT on real frames (`outputs/bench.txt`)

CDnet highway, 160×120 × 200 frames (19200×200), tol 1e-3, one BLAS thread:

| SVT | time | iterations | rank(L) | allocated | F1 | ‖L − L_exact‖/‖L‖ |
|---|---:|---:|---:|---:|---:|---:|
| exact Golub–Reinsch | 41.95 s | 19 | 102 | 1711 MB | 0.907 | — |
| randomized, w = 20 | **1.07 s** | 19 | 20 | **2.9 MB** | 0.908 | 2.5e-3 |
| randomized, w = 40 | 1.94 s | 19 | 34 | 6.7 MB | 0.908 | 1.5e-3 |
| randomized, w = 80 | 5.03 s | 19 | 56 | 16.7 MB | 0.907 | 9.0e-4 |

The randomized SVT is 39× faster with the same F1 on this window. The exact solver lets sensor noise into `L` once `1/μ` becomes small, and the width cap leaves it out. The width is not only a speed knob, though: it caps rank(L), which matters on other videos (see below).

A single 19200×200 SVD, measured separately: 70 ms randomized (w = 40), 259 ms LAPACK, 2151 ms our exact Golub–Reinsch. Exact Golub–Reinsch agrees with LAPACK to 7e-14.

### Choices made from measurements

On highway:
- **λ = 1/√m is the best of {0.5, 1, 2}×** (F1 0.834 / 0.861 / 0.821, before smoothing).
- **Tolerance 1e-3 instead of 1e-7** halves the iterations (39 → 19) with identical F1, so video runs use 1e-3. Tests and the synthetic grid keep 1e-7.
- **A 3×3 box filter on |S| before thresholding** raises F1 from 0.861 to 0.908, because it fills the hollow interiors of flat-coloured cars. Radius 2 and 3 were worse. The median baseline gets the same treatment.

### CDnet 2014, full videos, threshold calibrated on one video

Protocol:
- Every video is scored over its whole temporal ROI (11,334 frames in total) at 160 px wide. Batch RPCA solves consecutive 200-frame windows. Online RPCA sees each frame once, in order.
- Each method gets one threshold τ, chosen as the best F1 on **highway only**, then applied unchanged to the other four videos. The "oracle" number (best τ per video, using that video's ground truth) is shown alongside only to measure how well τ transfers.
- Every method scores the 3×3 box-smoothed |residual|.
- Ground truth: shadow pixels count as background; outside-ROI and unknown pixels are ignored.

Cells: oracle F1 / **F1 at the calibrated τ** (precision, recall). From `outputs/summary.md`:

| video | frames | stress | batch RPCA w=5 | batch RPCA w=20 | online RPCA | temporal median |
|---|---:|---|---:|---:|---:|---:|
| highway (calibration) | 1231 | none | 0.904 / **0.904** (P 0.86, R 0.95) | 0.899 / **0.899** (P 0.86, R 0.95) | 0.908 / **0.908** (P 0.87, R 0.95) | 0.909 / **0.909** (P 0.87, R 0.95) |
| backdoor | 1601 | shadows | 0.938 / **0.933** (P 0.96, R 0.91) | 0.946 / **0.944** (P 0.96, R 0.93) | 0.790 / **0.774** (P 0.68, R 0.90) | 0.872 / **0.843** (P 0.79, R 0.90) |
| badminton | 351 | camera shake | 0.785 / **0.784** (P 0.81, R 0.76) | 0.800 / **0.800** (P 0.81, R 0.79) | 0.861 / **0.861** (P 0.87, R 0.85) | 0.726 / **0.665** (P 0.57, R 0.80) |
| abandonedBox | 2051 | objects that stop | 0.462 / **0.462** (P 0.62, R 0.37) | 0.456 / **0.456** (P 0.61, R 0.37) | 0.747 / **0.520** (P 0.35, R 0.98) | 0.491 / **0.450** (P 0.53, R 0.39) |
| boats | 6100 | rippling water | 0.423 / **0.147** (P 0.09, R 0.50) | 0.357 / **0.162** (P 0.10, R 0.39) | 0.729 / **0.193** (P 0.11, R 0.88) | 0.544 / **0.182** (P 0.10, R 0.70) |

Calibrated τ: batch 0.080 (w=5), 0.065 (w=20); online 0.080; median 0.085. Speed: batch 113–293 frames/s (19 iterations per window), online 284–514 frames/s. The temporal median takes 22 ms per window.

What the table says:
- **Batch RPCA's threshold transfers; the median's doesn't.** At the highway τ, batch RPCA loses at most 0.005 F1 against its per-video oracle on backdoor, badminton and abandonedBox. On those same three videos the median loses 0.03–0.06, because its residual's scale changes with the scene.
- **Batch RPCA wins clearly where the background varies within a window:** shadows (0.944 vs 0.843) and camera shake (0.800 vs 0.665).
- **Online RPCA fixes the two failures batch RPCA can't:** objects that stop (oracle 0.747 vs 0.46) and the slow sailboat on boats (oracle 0.729 vs 0.36–0.42). It is also the best on camera shake (0.861), because its basis is fit over the whole video rather than 200 frames.
- **Online RPCA's weakness is drift.** It only learns the background from `x − s`, so any lasting change to the scene stays foreground. On backdoor its false-positive rate climbs from 0 to 1.6% over the video, and calibrated F1 falls to 0.774, below even the median. The same mechanism costs it precision on abandonedBox at the highway τ (0.35).
- **Boats is bad for everyone at a transferred τ.** 4,800 of its 6,100 frames contain no labelled motion, only rippling water, so every detection there is a false alarm.

### Batch vs online, side by side

`render <video> [frame]` writes `outputs/<video>[_frame].png` and `.mp4`. Top row: batch w=5; bottom row: online. Panels: input | `L` | 4·|S| | mask at the calibrated τ | ground truth (white = motion, grey = shadow, dark = ignored).

- `outputs/abandonedBox_1100.png`: the box has stood still for hundreds of frames. Batch has it in `L` and an empty mask there; online keeps it as foreground.
- `outputs/boats_5500.png`: batch's `L` contains a ghost of the slowly drifting sailboat; online's `L` is clean water and its mask covers the whole boat.
- `outputs/backdoor.png`: online's false positives at the left edge, where the scene changed after the warm start.

### Online RPCA settings (`outputs/online_sweep.md`)

γ and λ₂ were fixed at 0.999 and 1/√m before testing. The sweep is reported, not used to pick them. Each row gets its own τ calibrated on highway. Cells: calibrated F1 (oracle F1).

| γ | λ₂·√m | highway | backdoor | boats | badminton | abandonedBox |
|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 1 | 0.904 (0.904) | 0.868 (0.868) | 0.218 (0.714) | 0.862 (0.862) | 0.651 (0.684) |
| 0.995 | 1 | 0.909 (0.909) | 0.826 (0.827) | 0.208 (0.713) | 0.861 (0.861) | 0.601 (0.703) |
| **0.999** | **1** | 0.908 (0.908) | 0.774 (0.790) | 0.193 (0.729) | 0.861 (0.861) | 0.520 (0.747) |
| 1.0 | 1 | 0.907 (0.907) | 0.788 (0.794) | 0.169 (0.710) | 0.861 (0.861) | 0.499 (0.761) |
| 0.999 | 0.5 | 0.908 (0.908) | 0.791 (0.796) | 0.184 (0.715) | 0.861 (0.861) | 0.497 (0.761) |
| 0.999 | 2 | 0.907 (0.907) | 0.784 (0.796) | 0.220 (0.731) | 0.860 (0.861) | 0.565 (0.714) |
| 0.999 | 4 | 0.898 (0.898) | 0.836 (0.839) | 0.208 (0.728) | 0.858 (0.858) | 0.575 (0.690) |

This is the stability–plasticity tradeoff in numbers. Shorter memory (γ = 0.99) or a faster foreground leak (larger λ₂) absorbs scene changes sooner: backdoor improves to 0.868 and calibrated abandonedBox to 0.651. But the box is also absorbed sooner, so abandonedBox's oracle falls (0.761 → 0.684). Highway, the calibration video, scores 0.898–0.909 for every setting, so it cannot pick between them. Choosing γ per video would mean tuning on test ground truth.

### The rank cap is a modelling choice

On boats' last 1,000 frames a sailboat drifts slowly across. With up to 20 components, `L` can follow slow motion, so the boat ends up in the background: the worst window scores F1 0.20 at w = 20 and 0.50 at w = 2. A low cap hurts the other way on camera shake, where the background genuinely needs more modes:

| w | 2 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| boats, frames 5401–5600 | 0.50 | 0.45 | 0.39 | 0.25 | 0.20 |
| badminton, frames 1–200 | 0.78 | 0.82 | 0.85 | 0.87 | 0.87 |
| highway, frames 1–200 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 |

(oracle F1 per window). Both caps are reported above rather than picking one per video with the test ground truth.

### Tried and dropped

- **Adaptive threshold τ = c·σ̂**, with σ̂ the median-absolute-deviation noise level of each window's score and c calibrated on highway. It never beat the fixed τ for RPCA and collapsed on boats (recall ≤ 0.02). For the median it gained on boats (0.45) but lost badly on backdoor (0.69). It was removed.
- **400-frame windows** on boats' sailboat stretch helped both methods (RPCA 0.47 → 0.55 at w = 5), but the median more (0.57 → 0.71).

## What fails and why

- **Batch: objects that stop moving get absorbed into the background.** When abandonedBox's box stands still for a whole window, it is rank one, so it goes into `L`, and F1 on that stretch is about 0.04. The median fails the same way. Online RPCA fixes this, because it remembers the background from before the box arrived.
- **Batch: slow or large objects aren't sparse within a window.** A drifting sailboat or a tram is cheaper to explain as a low-rank change, so it leaves a ghost in `L`. Online RPCA fixes the sailboat.
- **Online: lasting scene changes stay foreground (drift).** The model learns only from `x − s`, a ±λ₂ sliver of anything it calls foreground, so a real change to the scene is absorbed slowly or not at all. A shorter memory trades this against forgetting stopped objects sooner (see the sweep). Telling "stopped object" apart from "changed scene" needs information neither model has, such as object identity or a rule like "absorb after T frames".
- **Online: per-frame lasso bias.** Each frame's `s` is shrunk by λ₂ and nothing removes that bias, unlike batch IALM where μ → ∞. On synthetic data the background error is 1.9e-3 (∝ λ₂), against 1e-7 for batch.
- **False alarms in empty scenes.** Both RPCA variants explain part of every frame as `S`. On thousands of frames of moving water with nothing labelled, those residuals become false positives. A detector meant for deployment would need a scene-level "is anything there" test or temporal persistence; neither is attempted here.
- **On a clean static scene, RPCA buys nothing.** On highway the 22 ms median matches every RPCA variant.
- **Flat object interiors look like background** at each pixel, which is why the 3×3 box filter is there. Larger radii smear the edges and lower F1.

## References

- E. Candès, X. Li, Y. Ma, J. Wright, *Robust principal component analysis?*, J. ACM 58(3), 2011.
- Z. Lin, M. Chen, Y. Ma, *The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices*, arXiv:1009.5055, 2010.
- J. Feng, H. Xu, S. Yan, *Online robust PCA via stochastic optimization*, NeurIPS 2013.
- N. Halko, P.-G. Martinsson, J. Tropp, *Finding structure with randomness*, SIAM Review 53(2), 2011.
- Y. Wang et al., *CDnet 2014: An expanded change detection benchmark dataset*, CVPR Workshops, 2014.
