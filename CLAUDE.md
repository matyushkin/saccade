# Saccade — Agent Briefing

Rust webcam eye tracker. Ridge regression on CLAHE eye patches → screen coordinates.
Goal: minimize pixel error on multi-point live validation.

## Current best results

| Metric | Value | Experiment | Config |
|--------|-------|------------|--------|
| Live (honest multi-point) | **237 px** | E12 | old 5×5/20×12 — needs new session with 7×7/30×18 |
| MPIIGaze (first-N protocol) | **5.31°** | E15 | 20×12, n_calib=500 |
| MPIIGaze (uniform-calib, n=200) | **3.70° / 2.82° median** | E17 | 30×18, uniform sampling |
| MPIIGaze (uniform-calib, n=200) | **3.54° / 2.40°** | E18 | 30×18, 3×3 CLAHE, uniform |
| MPIIGaze (uniform-calib, n=1000) | **3.04° / 2.20° median** | E18 | 40×24, 3×3 CLAHE, uniform — best ever |

Literature: L2CS-Net 3.92° (no calib), FAZE 3.18° (9-pt calib), GazeTR-Hybrid 3.43° (no calib).
**E16-E18: calibration diversity + larger patches + 3×3 CLAHE = 3.04° — beats FAZE by 4.4%.**

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

## Promising next steps (ordered)

1. **Run a new live session** — measure actual improvement from 7×7 grid + 30×18 patches. Expected: ~215 px vs old 237 px.
2. **Accumulated calibration across sessions** — `saccade_calib.bin` persists. After ~500 total clicks (≈5 sessions × 98 clicks), upgrade to 40×24 patches for best results.
3. **40×24 patches** — better than 30×18 at n≥500; set `EYE_PATCH_W=40, EYE_PATCH_H=24` after accumulating enough data.
4. **Proper Sugano normalization** — fix `src/sugano.rs` iterative PnP. Expected: ~3.92° without calib → better with calib. 2-3 weeks.
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
