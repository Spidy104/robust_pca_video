| video | stress | batch RPCA w=5 | batch RPCA w=20 | online RPCA | temporal median |
|---|---|---:|---:|---:|---:|
| highway (calibration) | none | 0.904 / **0.904** (P 0.86, R 0.95) | 0.899 / **0.899** (P 0.86, R 0.95) | 0.908 / **0.908** (P 0.87, R 0.95) | 0.909 / **0.909** (P 0.87, R 0.95) |
| backdoor | shadows | 0.938 / **0.933** (P 0.96, R 0.91) | 0.946 / **0.944** (P 0.96, R 0.93) | 0.790 / **0.774** (P 0.68, R 0.90) | 0.872 / **0.843** (P 0.79, R 0.90) |
| boats | rippling water | 0.423 / **0.147** (P 0.09, R 0.50) | 0.357 / **0.162** (P 0.10, R 0.39) | 0.729 / **0.193** (P 0.11, R 0.88) | 0.544 / **0.182** (P 0.10, R 0.70) |
| badminton | camera shake | 0.785 / **0.784** (P 0.81, R 0.76) | 0.800 / **0.800** (P 0.81, R 0.79) | 0.861 / **0.861** (P 0.87, R 0.85) | 0.726 / **0.665** (P 0.57, R 0.80) |
| abandonedBox | objects that stop | 0.462 / **0.462** (P 0.62, R 0.37) | 0.456 / **0.456** (P 0.61, R 0.37) | 0.747 / **0.520** (P 0.35, R 0.98) | 0.491 / **0.450** (P 0.53, R 0.39) |

Cells: oracle F1 (best τ for that video) / **F1 at the τ calibrated on highway** (precision, recall).
Calibrated τ: batch RPCA w=5 0.080, batch RPCA w=20 0.065, online RPCA 0.080, temporal median 0.085.
