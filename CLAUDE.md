# Saccade — Agent Briefing

Rust webcam eye tracker. Ridge regression on CLAHE eye patches → screen coordinates.
Goal: minimize pixel error on multi-point live validation.

## Current best results

| Metric | Value | Experiment | Config |
|--------|-------|------------|--------|
| Live (honest multi-point) | **237 px** | E12 | old 5×5/20×12 — needs new session with 7×7/30×18 |
| MPIIGaze (first-N protocol) | **5.31°** | E15 | 20×12, n_calib=500 |
| MPIIGaze (uniform, n=1000) | **3.04° / 2.20° median** | E18 | 40×24, 3×3 CLAHE — beats FAZE |
| MPIIGaze (uniform, n=2000) | 2.94° / 2.12° | E20 | 40×24, 3×3 CLAHE — 12/15 subjects |
| MPIIGaze (uniform, n=5000) | 2.83° / 2.04° | E20 | 40×24, 3×3 CLAHE — 10/15 subjects |

Literature: L2CS-Net 3.92° (no calib), FAZE 3.18° (9-pt calib), GazeTR-Hybrid 3.43° (no calib).
**Effective floor for ridge regression: ~2.8° (E20). Beating this requires CNN/Sugano.**

## DO NOT RETRY — dead ends

| Approach | Why failed | Retry condition |
|----------|-----------|-----------------|
| CNN without Sugano normalization (E3–E8) | 300–737 px, worse than pixels | Only with iterative PnP + debug crops showing correct warp |
| Smooth pursuit calibration (E11) | 329 px vs 142 px — saccades + lag corrupt labels | Only with velocity-based saccade filtering + per-sample weights |
| Head pose features appended to pixel ridge (E2) | 256 px, worse — features washed out by λ | Only with separate regression head |
| ResNet-50 gaze CNN (E7) | <2 FPS — UI unusable | Only on hardware with GPU / NPU |
| Patch size > 20×12 at n_calib=200 first-N (E15) | Plateau at 5.89-5.91° — but with uniform-calib 30x18 > 20x12 at ALL n | Always use uniform-calib when testing patch sizes |
| Zero-mean feature normalization (E12) | -3% improvement, not worth complexity | — |
| Decay sample weights (E12) | Neutral on this dataset, not worth complexity | Only for sessions with strong temporal drift |
| Horizontal flip of right eye (E16) | 6.12° vs 5.89° — worse; MPIIGaze normalized space is already consistent | — |
| Sobel-x gradient features (E16) | 5.87° vs 5.89° at n=200 — negligible; no gain at n=500 | — |
| Per-feature z-score normalization (E19) | 10.52° — catastrophic; near-constant pixels amplify noise when divided by tiny std | — |
| Separate per-eye regressors + average (E19) | 3.64° vs 3.54° — worse; loses binocular correlations between eyes | — |
| Head pose removal (E19) | 3.54° — neutral; head pose features contribute negligibly | — |
| Wide aspect ratio patches 40×12 (E19) | 3.61° — worse; 30×18 (5:3 ratio) better | — |
| Multi-scale features 40×24+20×12 (E21) | 3.06° at n=1000 vs 3.04° single-scale — neutral; marginal at n=500 | — |
| n_calib > 1000 for ridge (E20) | Curve flattening: n=1000→3.04°, n=5000→2.83° — only 0.21° gain for 5× more data | Only if accumulating many sessions passively |

## Promising next steps (ordered)

1. **Run a new live session** — measure actual improvement from 7×7 grid + 30×18 patches. Expected: ~215 px vs old 237 px.
2. **Accumulated calibration across sessions** — `saccade_calib.bin` persists. Plateau at ~n=1000 (≈10 sessions × 98 clicks). Beyond that, ridge regression hits its floor (~2.8°).
3. **40×24 patches after 500+ accumulated samples** — set `EYE_PATCH_W=40, EYE_PATCH_H=24` in `src/ridge.rs` after user accumulates 5+ sessions.
4. **Proper Sugano normalization** — fix `src/sugano.rs` iterative PnP. Estimated floor: 2.5° with calibration. 2-3 weeks.
5. **MediaPipe FaceMesh** — replace PFLD (68 pts) with 468-point model for better eye ROI.

## Key files

| File | Purpose |
|------|---------|
| `examples/webgazer.rs` | Main live app (pixel ridge) |
| `examples/webgazer_cnn.rs` | CNN variant with Sugano normalization |
| `examples/mpii_bench.rs` | MPIIGaze benchmark (`--patch WxH --n-calib N`) |
| `src/ridge.rs` | Ridge regressor + CLAHE feature extraction |
| `src/sugano.rs` | ETH-XGaze normalization (working but PnP is approximate) |
| `tools/preprocess_mpii.py` | MPIIGaze .mat → PNG+TSV converter |
| `EXPERIMENTS.md` | Full experiment log (read for details) |
| `results.jsonl` | Machine-readable benchmark numbers (one line/run) |

## Architecture

```
camera → rustface (face bbox) → PFLD (68 landmarks) → eye crop
→ CLAHE + bilinear resize to 20×12 → 480-D feature vector (×2 eyes + 3 head)
→ RidgeRegressor (LOO CV λ-tuning, n=45 clicks) → screen (x, y)
→ OneEuroFilter → cursor
```

## Benchmark commands

```sh
# Standard first-N protocol (reproducible baseline)
cargo run --release --example mpii_bench -- ./MPIIGaze_proc

# Best accuracy protocol (uniform calib, simulates structured grid)
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --patch 30x18 --n-calib 200 --uniform-calib
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --patch 30x18 --n-calib 500 --uniform-calib

# Absolute best (slow solve: 1923-D features)
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --patch 40x24 --n-calib 1000 --uniform-calib

# Query results
grep '"mpii_deg"' results.jsonl | jq '{exp,patch,n_calib,variant,mean_deg,median_deg}'
```
