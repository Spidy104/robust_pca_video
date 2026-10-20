# robust_pca_video

Background subtraction for fixed-camera video with robust PCA, written in Julia from the
equations up. The video is a matrix `X` with one grayscale frame per column. Robust PCA
splits it into a low-rank part `L` (the background) and a sparse part `S` (whatever moves):

```
minimize  ‖L‖_* + λ‖S‖₁   subject to   L + S = X,      λ = 1/√max(m, n)
```

There are two solvers:

- **Batch.** Inexact augmented Lagrange multipliers (Lin, Chen & Ma 2010) on 200-frame windows. Each iteration does one singular value thresholding step and one soft thresholding step. The SVT uses the randomized range finder and Golub–Reinsch SVD from my [svd_from_scratch](https://github.com/Spidy104/svd_from_scratch), so only a small w×200 matrix is ever decomposed exactly.
- **Online.** OR-PCA (Feng, Xu & Yan 2013), one frame at a time, with a rank-5 background basis and a forgetting factor. Each frame's small lasso problem is solved with FISTA.

Both are scored on five CDnet 2014 videos (11,334 frames) against the per-pixel ground truth, next to a plain temporal median as the baseline.

## Running it

Needs Julia 1.13, ffmpeg and curl. The only non-stdlib dependency is SVDScratch, which Pkg fetches from GitHub at a pinned commit.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl

scripts/fetch_cdnet.sh baseline highway              # ~3.5 min at 3 MB/s
scripts/fetch_cdnet.sh shadow backdoor
scripts/fetch_cdnet.sh dynamicBackground boats
scripts/fetch_cdnet.sh cameraJitter badminton
scripts/fetch_cdnet.sh intermittentObjectMotion abandonedBox

julia --project=scripts scripts/run.jl cdnet highway   # run highway first: it calibrates the thresholds
julia --project=scripts scripts/run.jl cdnet backdoor  # ...and the other four
julia --project=scripts scripts/run.jl summary         # outputs/summary.md
julia --project=scripts scripts/run.jl sweep           # online γ / λ₂ sensitivity
julia --project=scripts scripts/run.jl render boats 5500
julia --project=scripts scripts/run.jl bench           # exact vs randomized SVT
julia --project=scripts scripts/run.jl synthetic       # recovery on random problems
```

changedetection.net, the original CDnet host, no longer resolves, so `fetch_cdnet.sh` streams the category tarballs from the Hugging Face mirror [tauseefakhund/D2-CDNet](https://huggingface.co/datasets/tauseefakhund/D2-CDNet). It keeps one video and hangs up once the stream has moved past it. ffmpeg then converts the temporal ROI to raw 8-bit luma at 160 px wide. The ground-truth masks are resized with nearest neighbour so the label values survive.

## Code

`src/RobustPCA.jl` (about 330 lines) is the whole package: the batch and online solvers, SVT, soft thresholding, the median baseline (Wirth's selection), a 3×3 smoother, PR counting against CDnet labels, and raw video I/O. `scripts/run.jl` runs every experiment in this file and `test/runtests.jl` has the tests. The code has no comments, and no function takes more than four arguments (the field-by-field constructors of the two workspace structs aside); options live in small structs.

All large buffers are allocated once. Inside a batch iteration the only allocation is the SVD of the small w×200 core, about 0.15 MB at w = 20. The online `step!` allocates nothing, and neither do the fused batch update, the range finder, the median, the smoother or the PR tally; the tests check all of these with `@allocated`. Repeated solves leave the live heap flat (235 MB after each of five solves in `bench`).

The saved results were produced with OpenBLAS's default thread count, which is 6 here. A different count changes the last bits of the floating-point sums. I checked 2 threads against 6 and no F1 changed at three decimals.

## Results

### Synthetic recovery

4800×200 problems: `L0` of rank r with Gaussian factors, plus `S0` with ±1 at random positions. Batch solver, tolerance 1e-7. The table gives ‖L − L0‖/‖L0‖.

| rank \ density | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| 5  | 4e-7 | 8e-7 | 4e-7 | 1e-6 | 2e-6 |
| 10 | 3e-7 | 3e-7 | 5e-7 | 9e-7 | 5e-3 |
| 20 | 2e-7 | 3e-7 | 6e-7 | 4e-3 | 4e-2 |
| 30 | 2e-7 | 2e-7 | 2e-3 | 3e-2 | 0.15 |
| 40 | 7e-4 | 3e-3 | 1e-2 | 0.11 | 0.30 |

Recovery is exact up to the tolerance until rank and density get too large, as the theory says it should be.

### Exact vs randomized SVT

One 200-frame window of highway (19200×200), tolerance 1e-3, one BLAS thread, on an i5-1335U:

| SVT | time | rank(L) | allocated | F1 | ‖L − L_exact‖/‖L‖ |
|---|---:|---:|---:|---:|---:|
| exact Golub–Reinsch | 41.13 s | 102 | 1711 MB | 0.907 | |
| randomized, w = 20 | 0.83 s | 20 | 2.9 MB | 0.908 | 2.5e-3 |
| randomized, w = 40 | 1.72 s | 34 | 6.7 MB | 0.908 | 1.5e-3 |
| randomized, w = 80 | 4.90 s | 56 | 16.7 MB | 0.907 | 9.0e-4 |

Same F1, 50× faster at w = 20 and 24× at w = 40.

The width w is more than a speed setting, because it caps the rank of `L`. That matters later.

### CDnet 2014

The protocol is meant to keep the numbers honest. Every frame of each video's temporal ROI is scored. Each method gets a single threshold on its (3×3 box-smoothed) residual, chosen on highway and then used unchanged on the other four videos. Next to it is the oracle F1, the best threshold for that video picked with its own ground truth, which shows how much the fixed threshold costs. Shadow pixels count as background; outside-ROI and unknown pixels are skipped.

Each cell is oracle F1 / F1 at the highway threshold.

| video | frames | difficulty | batch w=5 | batch w=20 | online | median |
|---|---:|---|---:|---:|---:|---:|
| highway | 1231 | none (calibration) | 0.904 / 0.904 | 0.899 / 0.899 | 0.908 / 0.908 | 0.909 / 0.909 |
| backdoor | 1601 | shadows | 0.938 / 0.933 | 0.946 / 0.944 | 0.788 / 0.770 | 0.872 / 0.843 |
| badminton | 351 | camera shake | 0.785 / 0.784 | 0.800 / 0.800 | 0.861 / 0.861 | 0.726 / 0.665 |
| abandonedBox | 2051 | objects that stop | 0.462 / 0.462 | 0.456 / 0.456 | 0.747 / 0.524 | 0.491 / 0.450 |
| boats | 6100 | water, slow sailboat | 0.423 / 0.147 | 0.357 / 0.162 | 0.725 / 0.191 | 0.544 / 0.182 |

Precision and recall for every cell are in `outputs/summary.md`. Batch runs at roughly 240–630 frames/s and online at 1200–2200 frames/s. The median takes 22 ms per window.

No method wins everywhere.

Batch RPCA is best when the background changes within a window: shadows (0.944 against the median's 0.843) and camera shake. Its threshold also transfers. On backdoor, badminton and abandonedBox the highway threshold costs it at most 0.005 F1, while the median loses 0.03 to 0.06 because the scale of its residual changes from scene to scene.

Online RPCA fixes the two cases batch can't handle. A box that is left standing becomes part of a batch window's low-rank background, so batch scores 0.46 on abandonedBox. Online remembers the scene from before the box arrived and scores 0.75. A sailboat drifting slowly across boats leaves a ghost in the batch background, while online keeps it in the foreground. Online is also the best of the four on camera shake.

Online pays for this with drift. It learns the background only from `x − s`, so a lasting change to the scene is treated as foreground for a long time. On backdoor its false positive rate climbs from 0 to 1.7% over the video, and it ends up below the median.

Boats is bad for everyone once the threshold is fixed. 4,800 of its frames have nothing labelled in them, just rippling water, so any detection there is a false positive.

To see the difference, run `render <video> [frame]`. It writes an image and a 200-frame clip with batch on the top row and online below. The columns are input, `L`, 4·|S|, mask and ground truth.

- `outputs/abandonedBox_1100.png`: the box has been standing for hundreds of frames. Batch has absorbed it, online still marks it.
- `outputs/boats_5500.png`: the sailboat is ghosted into the batch background but sits cleanly in the online foreground.
- `outputs/backdoor.png`: online false positives on the left, where the scene changed after the start.

### Rank cap (batch)

Oracle F1 for single 200-frame windows:

| w | 2 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| boats, frames 5401–5600 | 0.50 | 0.45 | 0.39 | 0.25 | 0.20 |
| badminton, frames 1–200 | 0.78 | 0.82 | 0.85 | 0.87 | 0.87 |
| highway, frames 1–200 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 |

With a high cap, `L` can follow the slow sailboat. With a low cap, it can't follow the camera shake. Highway doesn't care. Both w = 5 and w = 20 are reported above instead of choosing a width per video with test labels.

### Online settings

γ = 0.999 and λ₂ = 1/√m were fixed before any results came in. This sweep shows how sensitive the method is to them; it was not used to choose them. Each row gets its own highway-calibrated threshold. Cells are calibrated F1 (oracle F1).

| γ | λ₂·√m | highway | backdoor | boats | badminton | abandonedBox |
|---:|---:|---:|---:|---:|---:|---:|
| 0.99 | 1 | 0.907 (0.907) | 0.866 (0.866) | 0.216 (0.699) | 0.862 (0.862) | 0.644 (0.681) |
| 0.995 | 1 | 0.909 (0.909) | 0.826 (0.827) | 0.209 (0.722) | 0.861 (0.861) | 0.602 (0.700) |
| 0.999 | 1 | 0.908 (0.908) | 0.770 (0.788) | 0.191 (0.725) | 0.861 (0.861) | 0.524 (0.747) |
| 1.0 | 1 | 0.907 (0.907) | 0.787 (0.794) | 0.176 (0.715) | 0.861 (0.861) | 0.499 (0.761) |
| 0.999 | 0.5 | 0.908 (0.908) | 0.790 (0.795) | 0.180 (0.712) | 0.861 (0.861) | 0.497 (0.762) |
| 0.999 | 2 | 0.906 (0.906) | 0.783 (0.795) | 0.220 (0.731) | 0.860 (0.861) | 0.568 (0.712) |
| 0.999 | 4 | 0.898 (0.898) | 0.835 (0.838) | 0.207 (0.728) | 0.858 (0.858) | 0.576 (0.690) |

Shorter memory absorbs scene changes faster, which helps backdoor (0.866 at γ = 0.99), but it also absorbs the stopped box faster (abandonedBox oracle 0.761 → 0.681). Highway scores almost the same for every setting, so the calibration video can't choose between them.

## Performance

These numbers come from profiling one 200-frame window of highway (19200×200) with the sampling profiler and small timers, on an i5-1335U. They were measured while rewriting the code, and the saved results were checked against the earlier ones afterwards.

In a batch iteration at w = 5, the updates of `S`, the residual, `Y` and `T` and the residual norm used to be five separate passes over 30 MB matrices. Together they cost 19.2 ms of a 28 ms iteration. They are now one pass that reads `X`, `L` and `Y` and writes `S`, `Y` and `T`, which takes 6.1 ms. The arithmetic per element is the same. The saved batch and median pixel counts match the earlier ones exactly on all 11,334 frames. These passes are limited by memory bandwidth, so replacing the two divisions by a reciprocal multiply gained nothing (6.14 against 6.17 ms) and I kept the divisions.

In the online step, soft thresholding is written with `copysign` and `max` so that it vectorizes, and the basis sweep is a matrix-vector product per column instead of a loop that strides across columns (263 µs down to 76 µs). The 5×5 Gram matrix took 78 µs with BLAS `syrk!`, which is a poor fit for a 19200×5 matrix, and takes 25 µs as a plain loop. The inner loop now stops at a relative change of 1e-4 instead of 1e-6. The old loop was capped at 50 iterations, and on boats it needed 78 on average to reach 1e-6, well above the cap, so many frames stopped there. FISTA's momentum cut the mean number of inner iterations by 8 to 24% across the five videos, and the slowest frame on abandonedBox went from 181 iterations to 54. On highway the online step went from 1.1 ms to 0.49 ms per frame.

The smoother and the tally were 99 µs and 194 µs per frame. The tally used a binary search over the thresholds, and now it computes the bin directly because the thresholds are evenly spaced. The smoother is fixed at 3×3 so its loops vectorize. Together they now take 11 to 39 µs per frame depending on the video, against about 290 µs before. They also run on one frame at a time instead of on window-sized score matrices, which saves about 245 MB.

Frames per second printed by `cdnet`, before and after. This laptop is noisy: one rerun of highway printed 345 frames/s for batch w = 5 where the table has 564, so read the ratios, not the absolute values.

| video | batch w=5 | batch w=20 | online |
|---|---:|---:|---:|
| highway | 260 → 564 | 149 → 257 | 390 → 1191 |
| backdoor | 193 → 567 | 113 → 244 | 402 → 1611 |
| badminton | 293 → 631 | 134 → 307 | 514 → 2188 |
| abandonedBox | 275 → 530 | 148 → 245 | 424 → 1256 |
| boats | 249 → 530 | 127 → 249 | 284 → 1278 |

Two changes that looked attractive did not survive a check. Skipping the power iteration in the range finder, or restarting it from the previous iteration's subspace, makes the solve faster but changes the answer: `L` moves by 0.2 to 9%, and on boats at w = 5 the F1 goes from 0.39 to 0.53 or 0.28. That is a different result, not the same result sooner, so the range finder stays as it was. For the BLAS thread count, the default was the fastest median on this machine (324 ms per w = 5 solve, against 375 ms with 2 threads and 445 ms with 1). Every setting has occasional runs twice as slow, which I put down to the mix of performance and efficiency cores, so `bench` uses one thread to keep its table steady.

What is left is the range finder. At w = 20 its two Gram–Schmidt passes in `orthonormalize!` take about 10 ms of a 35 ms iteration, and the matrix products take most of the rest. Speeding that up means changing svd_from_scratch, so I left it alone.

## Things that were tried

Some choices came from measurements on highway during development, before the options were fixed in code. λ = 1/√m beat 0.5× and 2× (0.861 against 0.834 and 0.821). Loosening the tolerance from 1e-7 to 1e-3 halved the iterations with no change in F1. A 3×3 box filter on |S| before thresholding filled in the hollow interiors of flat cars and lifted F1 from 0.861 to 0.908; larger filters smeared the edges.

Some ideas didn't work and were removed. An adaptive threshold at c times a MAD noise estimate per window never beat the fixed threshold for RPCA and fell apart on boats. Windows of 400 frames helped on the sailboat stretch, but they helped the median more than RPCA.

## Limitations

A batch window treats anything that stays still as background, so it can't keep a stopped object. Online keeps the object but also keeps every lasting scene change. Telling those two apart needs information that neither model has, such as object identity or a rule like "absorb after N frames".

Online RPCA shrinks each frame's foreground by λ₂, and nothing undoes that shrinkage. On the synthetic stream used in the tests this gives a background error of 1.9e-3, which halves when λ₂ is halved. Batch IALM stays below 1e-6 on the same kind of problem because its shrinkage disappears as μ grows.

Neither RPCA variant ever says that a frame contains nothing, so long empty stretches of moving water produce false positives. A deployed detector would need a scene-level test or temporal persistence.

On a clean static scene like highway, the 22 ms temporal median is as good as any of this.

## References

- E. Candès, X. Li, Y. Ma, J. Wright. Robust principal component analysis? J. ACM 58(3), 2011.
- Z. Lin, M. Chen, Y. Ma. The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv:1009.5055, 2010.
- J. Feng, H. Xu, S. Yan. Online robust PCA via stochastic optimization. NeurIPS 2013.
- N. Halko, P.-G. Martinsson, J. Tropp. Finding structure with randomness. SIAM Review 53(2), 2011.
- Y. Wang et al. CDnet 2014: An expanded change detection benchmark dataset. CVPR Workshops, 2014.

The frames under `outputs/` come from CDnet 2014. Please cite the paper above if you use them.
