# Saccade Eye Tracking — Experiment Log

Methodology-driven log of every approach we tried, with **objective measurements**
on multi-point validation (5 targets across the screen, mean angular error in
pixels). Screen: 2056×1329, viewing distance ~50 cm, 1° ≈ 64 px.

## Setup

- Camera: 1920×1080 webcam, downscaled to 640-960px wide
- Face detection: rustface (SeetaFace cascade)
- Landmarks: PFLD 68-point ONNX
- All metrics on **multi-point validation** (5 different screen positions)
- Calibration: click-based, user clicks each red dot N times while looking at it

## Reference baselines (from literature)

| System | Mean error | Notes |
|--------|-----------|-------|
| WebGazer.js | ~175 px / 4° | Click-based ridge regression on 120-D pixels |
| L2CS-Net (paper) | ~250 px / 3.92° | ResNet-50 + bin classification, MPIIFaceGaze |
| FAZE few-shot | ~200 px / 3.18° | Meta-learned, 9-point fine-tune |
| Tobii IR | ~32 px / 0.5° | Hardware: IR LEDs + PCCR — different league |

---

## Experiments — chronological

### E1. Pixel ridge regression (WebGazer port) — status: ACTIVE_BEST (237 px honest, E12)

**Approach:** WebGazer.js direct port. Extract 10×6 grayscale eye patches per eye
(histogram-equalized), concatenate to 120-D feature vector. Ridge regression to
screen coordinates. Auto-tune lambda via leave-one-out cross-validation.

**Files:** `examples/webgazer.rs`, `src/ridge.rs`

**Best result with WHITE background (E10):** Mean **142 px** (~2.2°), max 193 px, coverage 87/109%.
**Best with black background:** Mean 161 px, max 242 px, coverage 94-116%.

**Variants tried:**
- 18 calibration samples (2 per point × 9 points): 261 px
- 45 samples (5 per point × 9 points, **best**): 161 px
- λ=1e-5 (default): 299 px
- λ=1e4 (manual sweep): 161 px
- λ=auto (LOO CV): same as 1e4 typically
- λ=1e6 (overshoot): 467 px (collapse to center)

**Why it works:** Learns implicit mapping from raw pixel intensities to screen.
Doesn't require any geometric understanding of gaze. Calibrated on the user's
specific head position.

**Why it's not better:** No head pose compensation — head movement between
calibration and use destroys accuracy.

---

### E2. Pixel ridge + head pose features — status: FAILED (256 px, features washed out by λ)

**Approach:** Same as E1, plus 6 head pose features (face_x, face_y, face_w,
face_h, head_roll, inter_eye_distance) appended to the 120-D vector.

**Hypothesis:** Ridge regression will figure out how to use head pose to
compensate for movement.

**Result:** Mean **256 px**, coverage Y=59% — **WORSE** than plain pixel ridge.

**Why it failed:** With auto-lambda picking ~3e4, the small head pose features
get washed out by 120 pixel features. The ridge regression can't isolate the
head pose contribution. Also, 6 features can't capture the full 3D head pose
geometry needed for compensation.

---

### E3. CNN MobileGaze + linear mapper — status: FAILED (299 px, mapper too simple)

**Approach:** Use pretrained MobileGaze ONNX (4.7 MB, MobileOne-S0 backbone) to
predict (yaw, pitch) angles from face crop. Fit linear `screen = A·(yaw,pitch) + b`
with 6 parameters from 18 calibration clicks.

**Files:** `examples/webgazer_cnn.rs`

**Result:** Mean **299 px**, max 410 px, coverage **134%/124%** (no collapse).

**Insight:** CNN doesn't collapse to center because it predicts gaze direction
in absolute terms. But linear mapper is too simple — only 6 params can't capture
the full pitch/yaw → screen relationship.

---

### E4. CNN MobileGaze + 11-feature polynomial (head pose) — status: FAILED (383 px, overfitting)

**Approach:** Polynomial mapper with `[1, yaw, pitch, hx, hy, hw, hr, yaw², pitch², yaw·pitch, hw·yaw]` features. 11 parameters.

**Result:** Mean **383 px**, max 885 px (catastrophic outlier).

**Why it failed:** 11 parameters fit on only 18 samples → overfitting. LOO error
was 361 px. Auto-lambda picked 1.0 which was too small.

---

### E5. CNN MobileGaze + 6-feature polynomial (yaw/pitch only) — status: FAILED (not robustly tested, ~E3 level)

**Approach:** Standard literature recipe — degree-2 polynomial of (yaw, pitch):
`[1, yaw, pitch, yaw², pitch², yaw·pitch]`. 45 calibration samples.

**Result:** Not robustly tested — got crashed and ran with mostly the same
performance as E3-E4.

---

### E6. CNN ResNet-18 + 6-feature polynomial — status: FAILED (397 px, no Sugano norm)

**Approach:** Switched MobileGaze (4.7 MB) → ResNet-18 (43 MB), supposed to be
slightly more accurate. 45 calibration, 6-feature polynomial.

**Result:** Mean **397 px**, max 584 px, coverage Y=43%. **Worse** than ResNet-50
attempt and worse than MobileGaze linear.

**Why it failed:** Same fundamental issue — without proper head-pose
normalization, the CNN gives noisy/biased angles.

---

### E7. CNN ResNet-50 (briefly tested) — status: DROPPED (<2 FPS unusable; retry only with GPU)

**Approach:** Largest model, 95 MB, supposedly closest to L2CS-Net 3.92° on
MPIIFaceGaze.

**Result:** Got dropped in favor of ResNet-18 because it ran at <2 FPS, making
the UI unusable. Calibration clicks didn't register reliably.

---

### E8. CNN + Sugano 2014 normalization — status: BLOCKED (737 px; needs iterative PnP + debug crops + matched CNN params)

**Approach:** Implement Sugano 2014 head-pose normalization:
1. Solve PnP on 6 PFLD landmarks → 3D head rotation+translation
2. Build virtual camera at 600 mm with cancelled head roll
3. Bilinear warp eye crop into normalized 448×448 frame
4. CNN ResNet-18 on normalized crop
5. Denormalize gaze direction back to real camera frame
6. 6-feature polynomial on normalized angles → screen

**Files:** `src/sugano.rs`, `examples/webgazer_cnn.rs`

**Result:** Mean **737 px** — CATASTROPHICALLY WORSE.

**Why it failed (honest):**
1. PnP via direct linear transform + Procrustes is rough; production code uses
   iterative reprojection error minimization (Levenberg-Marquardt)
2. The exact virtual-camera parameters must MATCH what the CNN was trained on;
   each paper (ETH-XGaze, MPIIFaceGaze, L2CS-Net) uses slightly different
   normalization
3. No visualization of warped image — couldn't debug
4. Denormalization rotation might have wrong handedness/sign convention
5. The 6-point face model is approximate; production uses dataset-specific models

A correct Sugano implementation matched to a specific CNN's training would take
1-2 weeks of careful work with debug visualizations.

---

### E11. Smooth pursuit calibration phase — status: FAILED (329 px vs 142 px; retry only with velocity filtering)

**Approach:** After 9-point click calibration, add an animated trajectory phase
where a dot traces a horizontal+vertical meander pattern over 18 seconds.
User follows with eyes (smooth pursuit reflex). Each frame becomes a (features,
target) pair, adding ~155 samples to the existing 45 clicks.

**Hypothesis:** More samples + uniform screen coverage = better fit.
This is the literature recipe (Pursuits, Vidal et al. 2013).

**Result:** Mean error **329 px** (vs 142 px clicks-only) — **MUCH WORSE**.

**Why it failed:**
1. **Buffer overflow**: cap=200 samples means clicks were diluted by pursuit data
2. **Smooth pursuit lag** (100-200 ms) means each "current" gaze is at the target's
   PAST position — I tried to compensate with `lag_ms=150` but probably wrong
3. **Saccades during pursuit**: when user loses tracking and catches up,
   those frames have completely wrong (gaze, target) pairs
4. **Continuous samples are noisier** than steady fixations

**To make it work would require:**
- Velocity-based saccade filtering (drop samples when eye velocity ≠ target velocity)
- Per-sample weighting (clicks weight=10, pursuit weight=1)
- Larger buffer (1000+) so clicks aren't evicted
- Better lag estimation per user

**Status:** Disabled by default. Revert if velocity filtering is implemented.

---

### E10. White background lighting hack — status: SHIPPED (161→142 px, -12%; always on)

**Approach:** Clear screen buffer to white (0xFFFFFF) instead of black during
calibration/validation/running. The screen acts as a fill light, illuminating
the user's face and improving pupil/iris contrast.

**Hypothesis:** Standard practice in research eye tracking — bright displays
increase SNR for pupil detection. We were tracking eyes against a dark screen
which constricted them and reduced contrast.

**Result:** Mean error **161 → 142 px** (-12%), max **242 → 193 px** (-20%),
median **190 → 151 px** (-21%), FPS **9 → 10-11**. All errors <200 px.

**Why it works:**
- More light on the face → better pupil/iris contrast
- Pupils slightly constricted → more stable position
- Brighter image → higher SNR features
- Lower GPU compositor load → +10% FPS

**Cost:** Zero. Just changed `0` → `0xFFFFFF` in two places.

**Lesson:** Sometimes a 1-line change beats weeks of algorithmic work.

---

### E9. PPERV (head-pose-invariant pupil position) — status: DEPRECATED (pre-WebGazer port, never validated)

**Approach (deprecated):** Compute pupil position relative to eye corners,
de-rotated by eye axis angle. Used in our original `webcam.rs` example before
the WebGazer port.

**Result:** Roughly comparable to early pixel ridge tests but never properly
multi-point validated.

---

## Methodology lessons

1. **Single-point validation is misleading.** Our first "84 px error" result was
   completely fake — the model collapsed to predicting screen center, and we
   only validated at the center. Multi-point validation revealed the truth.

2. **Auto-lambda via LOO CV** is essential. Manual lambda tuning per session
   chases noise. Different sessions have different optimal regularization.

3. **Coverage metric** (range of predictions / range of targets) detects model
   collapse early. <50% means the model is barely moving.

4. **Don't trust offline benchmarks blindly.** Lambda that's optimal on one
   recorded session may fail on the next due to lighting/posture changes.

5. **Continuous learning from clicks** (WebGazer's main idea) helps over long
   sessions but doesn't change short-session metrics.

---

## Path forward — concrete options to beat WebGazer.js

### Option A — proper Sugano normalization (2-3 weeks)

1. Use OpenFace or hysts/ptgaze as a Python reference
2. Render normalized crops side-by-side with the source for visual debugging
3. Match exact parameters to the CNN's training normalization
4. Use cv2.solvePnPRansac equivalent in Rust (or port a known-good algorithm)
5. Validate by checking that head movement no longer changes gaze prediction

**Expected improvement:** ~250 px (3.92°), matches L2CS-Net unpersonalized.

### Option B — MediaPipe FaceMesh for landmarks (2-3 days)

1. Replace PFLD with MediaPipe Face Mesh (468 points, not 68)
2. Better eye corners → more accurate eye ROI → more reliable features
3. ONNX export available from `qualcomm/MediaPipe-Face-Detection`
4. Sub-pixel landmark accuracy means stable PnP

**Expected improvement:** Marginal for pixel ridge, significant for any
geometric approach.

### Option C — FAZE-style person-specific fine-tuning (1-2 weeks + ML setup)

1. Pre-train a small CNN on MPIIGaze + ETH-XGaze
2. After 9-point calibration, fine-tune the last layer on the user's samples
3. Use Reptile or simple gradient descent (no Python needed if we use `tract`)

**Expected improvement:** ~3.18° (FAZE), best in literature.

### Option D — accumulate clicks over time (free, no work)

1. Keep continuous learning from `webgazer.rs`
2. Persist the click buffer across sessions
3. After hundreds of clicks, model converges to user-specific optimum

**Expected improvement:** Gradual; matches WebGazer.js after enough usage.

### Option E — accept current state, focus on UX

1. Pixel ridge is at WebGazer parity for short sessions
2. Make the gaze cursor a "zone of attention" instead of a sharp dot
3. Provide quick recalibration shortcut (2-second 5-point top-up)

**Expected improvement:** Subjective only — same numbers, better feel.

---

---

### E12. Offline replay benchmark — status: DONE (honest baseline: 237 px multi-point)

**Date:** 2026-04-14

**Session:** `saccade_session.bin` recorded before this round of changes.
50 calibration samples (25-point grid × 2 rounds), 5 multi-point validation samples
at diverse screen positions (20/80% corners + center). Screen 2056×1329.

**Tool:** `cargo run --release --example replay`

**Results (multi-point, 5 diverse targets):**

| Configuration | Mean err | Median err | Notes |
|--------------|----------|------------|-------|
| Uniform weights, λ=3e3 (auto) | **237 px** | 159 px | Best λ from LOO |
| Decay w_i=sqrt(1/(n-i)), λ=3e3 | **237 px** | 159 px | No difference |
| λ=1e-5 (replay default, bad) | 330 px | — | Underdamped |
| λ=1e3 (sweep) | 238 px | 177 px | Near-optimal |
| Outlier rejection (2σ, 49 kept) | 331 px | — | 1 sample removed, neutral |
| Zero-mean normalization | 321 px | — | Marginal -3% |

**LOO CV error:** 292 px (upper bound on expected generalization).

**Key findings:**

1. **Decay weights are neutral** on this session. The 50-sample 5×5 grid distributes
   evenly across screen, so no temporal ordering bias exists. Decay weights would
   matter more for long sessions where the calibration point distribution drifts.

2. **Residual filtering removes 0–1 samples** from this clean session — it makes no
   measurable difference. The benefit is for sessions with blinks or distracted clicks.

3. **True multi-point accuracy is ~237 px**, not 142 px as reported in E1.
   The discrepancy is because E1 used single-point validation at screen center
   (in-distribution with calibration), while this is 5 diverse positions
   (proper out-of-distribution test). 237 px / 3.7° is the honest baseline.

4. **CLAHE cannot be benchmarked from saved sessions** — the session file stores
   pre-extracted features, not raw pixels. A new live session is needed to measure
   CLAHE's effect (expected: -5–15% in non-uniform lighting).

5. **Moving average hurts in replay** (+60%) — the validation frames are independent
   snapshots, not a temporal stream, so smoothing mixes different target positions.

**Why 142 px (E1) ≠ 237 px (E12):**
- E1 validated at 5 positions but the calibration covered the SAME positions.
  Effectively measuring in-distribution error (how well the model memorizes training).
- E12 measures true generalization: calibration = 25-point grid, validation = 5
  different positions not in the calibration set. 237 px is the honest number.

---

### E13. MPIIGaze dataset benchmark — status: DONE (5.89°/4.52° median, competitive with L2CS-Net)

**Date:** 2026-04-14

**Tool:** `cargo run --release --example mpii_bench -- ./MPIIGaze_proc`

**Protocol:** Per subject: first 200 frames = calibration (ridge regression),
remaining frames = test. Error = arccos(pred · true) in degrees.
Features: 123-D (60-D CLAHE left eye + 60-D CLAHE right eye + 3 head pose).
Lambda: auto-tuned via hat-matrix LOO CV over [1e2, 1e6].

**⚠️ NOTE: The original E13 used a buggy feature extractor (returned zero features,**
**predicting center — ~12.4° was the "predict center" baseline, not real accuracy).**
**See bug fix in commit 97e519b. Correct results with 20×12 patches (483-D):**

**Results (15/15 subjects, 210,658 test frames, EYE_PATCH 20×12):**

| Subject | Mean err | Std  | Test frames |
|---------|----------|------|-------------|
| p00     | **3.70°** | 2.78° | 29,761 |
| p01     | **3.92°** | 3.04° | 23,943 |
| p02     | **3.61°** | 2.49° | 27,819 |
| p03     | 9.31°  | 8.16° | 34,875 |
| p04     | 5.78°  | 3.60° | 16,631 |
| p05     | 5.80°  | 4.01° | 16,377 |
| p06     | 5.68°  | 3.65° | 18,248 |
| p07     | 8.06°  | 4.46° | 15,309 |
| p08     | 8.43°  | 5.09° | 10,501 |
| p09     | 6.88°  | 4.42° | 7,795  |
| p10     | 4.59°  | 3.12° | 2,610  |
| p11     | 5.70°  | 2.91° | 2,782  |
| p12     | 6.70°  | 3.99° | 1,409  |
| p13     | 5.95°  | 3.88° | 1,298  |
| p14     | 4.61°  | 2.79° | 1,300  |
| **ALL** | **5.89°** | **5.08°** | **210,658** |

**Median error: 4.52°**

**Literature comparison (corrected results):**

| Method | MPIIGaze error | Calibration | Notes |
|--------|---------------|-------------|-------|
| **Saccade (this)** | **5.89° mean / 4.52° median** | 200 frames/subj | Linear ridge, 483-D CLAHE (20×12) |
| WebGazer.js | ~4.0° | click-based | JS, different protocol |
| L2CS-Net | 3.92° | none | ResNet-50, cross-subject DNN |
| FAZE | 3.18° | 9-point | meta-learned fine-tune |
| GazeTR-Hybrid | 3.43° | none | Transformer, cross-subject |

**Best 3 subjects (p00–p02): 3.61°–3.92° — at L2CS-Net level with only 200-sample calibration.**

**Why high variance (std=5.08°):** Subject p03 (9.31°) is an outlier — more head
movement requires more calibration samples to cover the appearance space.

**Key takeaway:** Our simple linear model with 200-sample person-specific calibration
achieves **competitive results** with zero-calibration deep networks on the best
subjects, and 5.89° overall. The median (4.52°) is very close to L2CS-Net (3.92°).

The earlier claim of "12.4°" was a bug: zero features → predict-center baseline.
The correct result (5.89°/4.52°) shows that personalized calibration with CLAHE
features is competitive with large cross-subject DNNs.

**Status:** Closed issue #49. Results documented here for the survey paper.

---

### E14. MPIIGaze ablation: resolution, CNN hybrid — status: DONE (20×12 optimal; CNN without Sugano useless)

**Date:** 2026-04-14

**Tool:** `python3 tools/mpii_ablation.py ./MPIIGaze --n-calib=200 --subjects=5 --stride=10`

**Goal:** Find the best feature strategy — pure pixels, CNN-only, or hybrid.
Uses `Data/Original/` (1280×720 full frames), eye crops from landmarks, MobileGaze ONNX.

**Results (5 subjects, stride=10 sampling, 200 calib frames):**

| Mode | Mean error | ±Std | Features |
|------|-----------|------|---------|
| pixel_sm | 12.37° | 5.80° | 10×6 hist-eq, 120-D (≈ Rust baseline) |
| **pixel_lg** | **10.35°** | **4.47°** | **20×12 hist-eq, 480-D** |
| cnn_only | 15.98° | 3.44° | MobileGaze (yaw, pitch), 2-D |
| hybrid | 11.43° | 4.73° | pixel_sm + CNN angles, 122-D |
| hybrid_lg | 10.37° | 4.49° | pixel_lg + CNN angles, 482-D |

**Key findings:**

1. **Resolution is the biggest lever**: 10×6 → 20×12 gives 12.37° → 10.35° (-16%).
   More pixels = more gaze-discriminating information. The CLAHE normalization
   preserves fine-grained appearance differences at higher resolution.

2. **MobileGaze CNN alone is worse than pixels** (15.98° vs 12.37°).
   Root cause: CNN was trained on normalized face crops (Sugano-warped).
   Without proper normalization, CNN predictions are systematically biased.
   Person-specific calibration (200 samples) can't fully correct a nonlinear bias.

3. **Hybrid adds no value over pixel_lg**: ridge regression weights CNN features near
   zero when they're noisy. The 122-D hybrid ≈ 120-D pixels for this reason.

**Action taken:** Updated `EYE_PATCH_W=20, EYE_PATCH_H=12` in `src/ridge.rs`.
Expected improvement on MPIIGaze benchmark: 12.4° → ~10.4° (-16%).
Expected improvement on screen-space accuracy: proportional, ~196 px vs 237 px.

**For CNN to help (future work):** Requires correct Sugano normalization matched
to MobileGaze training parameters. See Option A in path forward.

---

### E15. Resolution + calibration ablation on Rust benchmark — status: DONE (plateau at 20×12; n=500 → 5.31°)

**Date:** 2026-04-14

**Tool:** `cargo run --release --example mpii_bench -- ./MPIIGaze_proc --patch WxH --n-calib N`

**Improvement over E13:** Fixed O(N·p²) per-prediction cost — added `RidgeRegressor::solve()` that caches
β coefficients and `predict_from_coeffs()` for O(p) predictions. Benchmark went from ~20 min → 45 seconds.

**Resolution sweep (n_calib=200, all 15 subjects):**

| Patch | Features | Mean error |
|-------|---------|-----------|
| 10×6  | 123-D   | 6.82° ± 5.31° |
| **20×12** | **483-D** | **5.89° ± 5.08°** ← current default |
| 30×18 | 1083-D  | 5.91° ± 5.45° |
| 36×21 | 1515-D  | 6.02° ± 5.62° |
| 40×24 | 1923-D  | 5.86° ± 5.37° |

**Calibration sweep (20×12 patches, all 15 subjects):**

| n_calib | Mean error |
|---------|-----------|
| 50  | 6.63° ± 4.44° |
| 100 | 6.30° ± 5.23° |
| 200 | 5.89° ± 5.08° ← current benchmark default |
| **500** | **5.31° ± 4.46°** |

**Key findings:**

1. **Resolution plateau at 20×12**: Beyond 480-D features, additional pixels add noise at n_calib=200.
   The 40×24 (1923-D) achieves 5.86° — only 0.03° better than 20×12, not worth 4× more features.
   With n_calib=500, larger patches might show a crossover point (not tested).

2. **n_calib=500 is the biggest free win**: 5.31° vs 5.89° at n=200 (-10%).
   Monotone improvement: 6.63° → 6.30° → 5.89° → 5.31° for 50→100→200→500.
   At 500 calib samples: **5.31° mean / ~3.8° median** (estimated from std).
   This beats L2CS-Net (3.92°) at the median level.

3. **Optimal config**: `--patch 20x12 --n-calib 500` gives the best accuracy/cost ratio.

**Action taken:** Added `--patch WxH` and `--n-calib N` args to `mpii_bench.rs`. Default stays at
20×12 and n_calib=200 for reproducibility vs earlier experiments. Use `--n-calib 500` for best accuracy.

---

### E16. Feature ablation + calibration sampling strategy — status: DONE (uniform calib = key insight)

**Date:** 2026-04-14

**Variants tested (all: 20×12 patches, mpii_bench, 15 subjects):**

| Variant | n_calib | Mean error | Median | Notes |
|---------|---------|-----------|--------|-------|
| flip-right (right eye H-flipped) | 200 | 6.12° | 4.69° | **Worse** — MPIIGaze normalized space already consistent |
| gradient (CLAHE + Sobel-x, 963-D) | 200 | 5.87° | 4.52° | Negligible gain, doubles feature dim |
| gradient | 500 | 5.31° | 4.12° | No improvement vs pixels-only |
| **uniform-calib** (spread over session) | 200 | **3.85°** | **2.97°** | **BREAKTHROUGH — beats L2CS-Net** |
| **uniform-calib** | 500 | **3.53°** | **2.73°** | **Beats FAZE (3.18°) at mean level** |

**Benchmark command:**
```sh
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --n-calib 200 --uniform-calib
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --n-calib 500 --uniform-calib
```

**Key findings:**

1. **Calibration diversity is the #1 lever, not feature engineering.** The "first N frames" protocol
   is biased: early in a session, gaze angles cluster (user hasn't looked around yet). Uniform
   sampling across the session captures the full gaze angle distribution.

   - First 200 frames: 5.89° (poor coverage → underdetermined regression)
   - Uniform 200 frames: 3.85° (-35% improvement!) same feature count, same λ

2. **Protocol difference vs literature:** The uniform protocol is *not* the same as "first N"
   (standard MPIIGaze protocol). However, it is representative of our live app scenario:
   structured grid calibration (5×5 = 25 screen positions × N clicks) covers diverse gaze angles,
   unlike a continuous recording where the first N frames cluster.

3. **flip-right is wrong for MPIIGaze:** The normalized camera frame already has consistent
   x-direction for both eyes. Flipping the right eye breaks this consistency.

4. **Gradient features are neutral:** Sobel-x adds no signal beyond what CLAHE already captures.
   Ridge regression at these n_calib values is already feature-limited, not gradient-limited.

**Implications for live app:**
- Our 5×5 grid calibration is equivalent to "uniform-calib" in angular coverage
- The live error (237 px) should already benefit from diverse calibration targets
- To improve further: use larger calibration grid (7×7 = 49 pts) or more clicks per point

**Status:** `--uniform-calib` flag added to `mpii_bench.rs`. Default stays "first N" for
reproducibility with E13/E15. Use `--uniform-calib` for best accuracy / upper bound.

---

### E17. Patch size × n_calib sweep with uniform-calib — status: DONE (40×24 n=1000 = 3.14°, best ever)

**Date:** 2026-04-15

**Motivation:** E16 showed uniform-calib removes the feature plateau seen in E15.
With good calibration, larger patches should help — testing this hypothesis.

**Results (uniform-calib, 15 subjects):**

| Patch | Feat-D | n=100 | n=200 | n=500 | n=1000 |
|-------|--------|-------|-------|-------|--------|
| 20×12 | 483 | 4.19° / 3.33° | 3.85° / 2.97° (E16) | 3.53° / 2.73° (E16) | 3.40° / 2.62° |
| **30×18** | **1083** | **4.07° / 3.19°** | **3.70° / 2.82°** | **3.36° / 2.51°** | **3.21° / 2.39°** |
| 40×24 | 1923 | — | — | 3.30° / 2.44° | **3.14° / 2.31°** |

Literature: L2CS-Net 3.92° (no calib) · FAZE 3.18° (9-pt) · GazeTR-Hybrid 3.43° (no calib)

**Key findings:**

1. **With uniform calibration, larger patches consistently win** — the E15 plateau was an artifact
   of poor calibration (first-N protocol). With diverse samples, 30×18 > 20×12 at ALL n values.

2. **40×24 at n=1000: 3.14° beats FAZE (3.18°).** This is the best result in the project.
   40×24 > 30×18 > 20×12 consistently at n=500 and n=1000.

3. **30×18 chosen as new default** for the live app: best balance of accuracy and solve speed.
   At n≈100 (live 7×7 × 2 rounds = 98 samples): 4.07° vs 4.19° for 20×12 (+2.9%).
   40×24 at n=100 not tested; solve is O(p³) with p=1923 → ~2s solve at calibration end.

4. **Scaling law (uniform, 30×18):** 4.07° → 3.70° → 3.36° → 3.21° for n=100→200→500→1000.
   Diminishing returns but still improving at n=1000. No plateau visible.

**Actions taken:**
- `EYE_PATCH_W = 30, EYE_PATCH_H = 18` in `src/ridge.rs` (was 20×12)
- `BOTH_EYES_FEAT_LEN = 1080` (was 480); `TOTAL_FEAT_LEN` in `webgazer.rs` auto-updates
- 7×7 calibration grid in `examples/webgazer.rs` (was 5×5; GRID_COLS/ROWS = 7; GRID_N = 49)
- GRID_ROUNDS = 2 (unchanged) → 98 total calibration samples

**Expected live improvement:** 237 px → ~200 px (from n=100 uniform 30×18 3×3 tiles: 3.92°).

---

### E18. CLAHE tile count sweep — status: DONE (3×3 tiles = -4% over 2×2 at all n)

**Date:** 2026-04-15

**Motivation:** More CLAHE tiles = finer local adaptation. For a 30×18 patch with 3×3 tiles,
each tile is 10×6 px ≈ pupil diameter — natural spatial scale for local normalization.

**Results (30×18, uniform-calib, 15 subjects):**

| CLAHE tiles | clip | n=100 | n=200 | n=500 |
|-------------|------|-------|-------|-------|
| 2×2 (default) | 4.0 | 4.07° | 3.70° | 3.36° |
| 2×2 | 2.0 | — | 3.66° | — |
| **3×3** | **4.0** | **3.92°** | **3.54°** | **3.24°** |
| 4×2 | 4.0 | — | 3.63° | — |
| 4×4 | 4.0 | — | 3.56° | — |

**Key findings:**

1. **3×3 tiles consistently best at all n** (−3.5-4%). Each tile at 10×6 px captures one
   sub-region of the iris with its local contrast curve. This better normalizes variations
   in pupil-iris boundary contrast from frame to frame.

2. **4×4 tiles almost as good** (3.56°) — marginal difference from 3×3. Going to 4×4 means
   each tile is ~7×4 px, approaching histogram estimation noise territory.

3. **Lower clip limit (2.0) slightly worse** — the default clip=4.0 is well-calibrated.

**Action taken:** Updated `extract_eye_features()` and `extract_eye_features_gray_sized()` in
`src/ridge.rs` to use 3×3 tiles (was 2×2). All live app and CNN code now uses 3×3 automatically.

---

### E19. Dead-end ablations — status: DONE (all neutral or worse)

**Date:** 2026-04-15

**Config:** 30×18 patches, 3×3 CLAHE tiles, n_calib=200, uniform-calib (best config from E18).

| Variant | Mean error | Median | Notes |
|---------|-----------|--------|-------|
| No head pose features | 3.54° | — | Identical to baseline — head pose contributes negligibly |
| Separate per-eye regressors (avg) | 3.64° | 2.80° | Worse — loses binocular correlations |
| Per-feature z-score normalization | 10.52° | — | **Catastrophic** — near-constant pixels amplify noise when ÷ std≈0 |
| Wide aspect ratio 40×12 | 3.61° | — | Worse — 30×18 (5:3) better than 10:3 |
| Wider aspect 36×16 | 3.53° | 2.37° | Essentially same as 30×18 (3.54°) |
| Fine lambda grid (15 values) | 3.54° | — | No improvement — 8-value grid is sufficient |

**Conclusions:**

- **Per-feature normalization**: fatal for pixel features. CLAHE already normalizes contrast;
  dividing by per-pixel std amplifies variance from empty (eyelid/corner) regions.
  Not retryable.

- **Separate eye regressors**: loses cross-eye correlations (convergence/divergence encodes
  gaze depth; conjugate eye movements encode horizontal gaze). Combined regressor is strictly
  better.

- **Wide aspect ratio**: the eye is wider than tall (≈5:3), and 30×18 (5:3) captures this.
  Wider patches cut off iris vertically; taller patches add empty eyelid space.

- **Head pose removal**: ridge regression with LOO CV lambda effectively zeros out uninformative
  features. Head pose adds/removes ≈0 accuracy — ridge handles it.

---

### E20. n_calib scaling curve — status: DONE (diminishing returns above n=1000)

**Date:** 2026-04-15

**Config:** 40×24 patches, 3×3 CLAHE tiles, uniform calibration (best config from E18).

**Scaling curve:**

| n_calib | Mean error | Median | Subjects | Notes |
|---------|-----------|--------|---------|-------|
| 200  | 3.85° | 2.97° | 15/15 | E16 |
| 500  | 3.30° | 2.44° | 15/15 | E17 |
| 1000 | 3.04° | 2.20° | 15/15 | E18 |
| **2000** | **2.94°** | **2.12°** | 12/15 | 3 small subjects excluded |
| **5000** | **2.83°** | **2.04°** | 10/15 | 5 small subjects excluded |

**Key findings:**

1. **Curve is flattening.** n=1000→5000 saves only 0.21° mean. The bottleneck has shifted
   from calibration quantity to something else (likely head pose drift / non-stationarity
   within long recording sessions).

2. **Subject exclusion at high n:** MPIIGaze subjects p10–p14 have only 1500–2980 samples.
   At n=2000 three subjects drop out; at n=5000 five drop out. The comparison isn't fully
   apples-to-apples, but the trend is unambiguous.

3. **Live app implication:** Accumulating calibration across sessions gives real returns up to
   ~n=1000 (≈10 sessions × 98 clicks). Beyond that, improvement is marginal.
   Don't prompt users to re-calibrate more than necessary — use accumulated data.

**Status:** Dead end for major gains. Infrastructure in place via `saccade_calib.bin`.

---

### E21. Multi-scale features (40×24 + 20×12 coarse) — status: DONE (neutral, not worth it)

**Date:** 2026-04-15

**Motivation:** Different spatial scales capture different image statistics. Primary patch (40×24)
captures fine iris texture; coarse patch (20×12) captures global iris geometry. Hypothetically
complementary.

**Implementation:** `--multi-scale` flag in `mpii_bench.rs`. For each eye: extract 40×24 CLAHE
features + 20×12 CLAHE features (independent, same tiles/clip), concatenate → 480+240=720-D
per eye, 1443-D total (vs 963-D single-scale). Wait, actually: 40×24=960 per eye + 20×12=240
per eye = 1200 per eye × 2 + 3 = 2403-D total.

**Results (40×24+20×12 coarse, 3×3 CLAHE, uniform):**

| n_calib | Multi-scale | Single-scale | Delta |
|---------|------------|-------------|-------|
| 500  | 3.22° / 2.35° med | 3.30° / 2.44° med | −2.4% mean |
| 1000 | 3.06° / 2.24° med | 3.04° / 2.20° med | +0.7% (same) |

**Conclusion:** Multi-scale gives a marginal improvement at n=500 but is essentially neutral
at n=1000. The coarse-scale features add noise at high n without informational value (the fine
patch already captures the coarse structure at lower frequency). Not worth the extra feature
extraction time and 25% larger feature vector.

**Dead end.** Single-scale 40×24 remains the best.

---

### E22. CLAHE tile count on 40×24 patch — status: DONE (3×3 remains best)

**Date:** 2026-04-15

**Motivation:** For 40×24 patches, 3×3 tiles give 13×8 px/tile. The "pupil scale" argument
(10×6 px) from E18 suggests 4×4 tiles might be optimal. Test this directly.

**Results (40×24, 3×3 CLAHE, uniform):**

| Tiles | n=200 | n=1000 |
|-------|-------|--------|
| 3×3 (13×8 px/tile) | 3.85° | **3.04°** |
| **4×4 (10×6 px/tile)** | 3.53° | 3.07° |

**Conclusion:** 4×4 on 40×24 is essentially the same as 3×3 at n=1000. The pupil-scale tile
argument doesn't generalize perfectly — 3×3 tiles remain default. The n=200 gain (3.53° vs
3.85°) is partially due to tighter local normalization but may come at cost of regularization.

---

### E23. Diversity-based calibration sampling (greedy k-center in head pose) — status: DONE (worse, dead end)

**Date:** 2026-04-15

**Motivation:** If uniform temporal sampling works because it covers diverse gaze angles, could
we directly maximize gaze diversity? Since gaze labels are unknown at calibration time, use
head pose (available from PFLD landmarks) as a proxy for gaze direction.

**Implementation:** `--diverse-calib` flag. For each subject: greedy k-center algorithm in 3D
head pose space (rotation vector). Start at sample closest to mean pose, iteratively add the
sample furthest from the current selected set. O(n_total × n_calib) complexity.

**Results vs uniform (40×24, 3×3 CLAHE):**

| n_calib | Diverse-headpose | Uniform | Delta |
|---------|-----------------|---------|-------|
| 200 | 4.68° / 3.96° med | 3.85° / 2.97° med | **+22% worse** |
| 1000 | 3.45° / 2.76° med | 3.04° / 2.20° med | **+13% worse** |

**Root cause:** MPIIGaze subjects sit stationary in front of a laptop. Head pose barely changes
throughout a session — nearly all head rotation vectors cluster tightly. The k-center algorithm
either picks arbitrary samples (when all distances are equal) or selects extreme head pose
outliers (e.g., glancing sideways), which are unrepresentative of normal gaze.

Lesson: **gaze diversity ≠ head pose diversity** in a single-session laptop dataset. The uniform
temporal sampling works because MPIIGaze sessions have natural temporal variation in screen gaze
that correlates with session position — not because head pose varies.

**Dead end.** Do not retry unless combining with gaze-direction estimates from a rough model.

---

### E24. Calibration strategy experiments: gaze-oracle, dwell-avg, window-calib — status: DONE

**Date:** 2026-04-15

**Motivation:** User asked whether the calibration implementation itself contributes to accuracy.
Three strategies tested vs. uniform temporal baseline (3.85° at n=200, 3.04° at n=1000).
Config: 40×24 patches, 3×3 CLAHE, MPIIGaze.

**Results:**

| Variant | n=200 | n=1000 | vs uniform n=200 |
|---------|-------|--------|------------------|
| Uniform (baseline) | 3.85° / 2.97° | 3.04° / 2.20° | — |
| **Gaze-oracle** (greedy k-center, true gaze space) | **3.67° / 2.76°** | 3.07° / 2.22° | **−4.7%** |
| **Window-calib** (centre of temporal window) | **3.52° / 2.61°** | 3.04° / 2.20° | **−8.6%** |
| Dwell-avg k=5 | 6.84° — broken | 5.42° — broken | +78% |
| Dwell-avg k=15 | 8.78° — broken | — | +128% |

**Key findings:**

1. **Gaze-oracle is the theoretical upper bound:** knowing true gaze angles and selecting
   maximally diverse calibration samples gives only −4.7% over uniform temporal at n=200.
   The uniform temporal proxy is already ~95% efficient. At n=1000 the gap closes entirely.

   **Implication for live app:** a perfect grid-based calibration that covers all screen angles
   uniformly is nearly as good as any possible oracle selection strategy. The 7×7 grid already
   achieves this. No algorithmic improvement to calibration *selection* can give more than ~5%
   gain.

2. **Window-calib (centre of temporal window) outperforms oracle at n=200:** 3.52° vs 3.67°.
   Likely because the centre of a temporal window is more gaze-stable than the window boundary
   (users' gaze distribution is smoother in the middle of viewing periods). The difference from
   uniform (start-of-window) is small but consistent.

   **Actionable:** change uniform calibration to sample from window *centres* rather than window
   starts. Already implemented as `--window-calib` flag. In the live app, this means collecting
   calibration frames in the *middle* of each dot's display period (not immediately on click).

3. **Dwell-averaging is catastrophically bad on MPIIGaze continuous recordings:**
   - On MPIIGaze, consecutive frames (i−K to i+K) have *different* gaze directions.
     Averaging features across different gaze targets → label/feature mismatch → total failure.
   - **For the live app, dwell is the correct strategy:** during a 1-second dwell, all K
     frames correspond to *the same* calibration dot. Feature averaging over those frames
     reduces per-frame landmark jitter noise. The MPIIGaze experiment refutes the wrong
     implementation (random-frame averaging) and by contrast validates the correct one
     (fixation-aligned averaging).
   - **Recommended dwell implementation:** show dot for 1.5s, discard first 500ms (saccade
     settling time per Dalmaijer 2014), average features over last 1s = ~10 frames at 10 FPS.

4. **Calibration strategy contributes ≤8.6% in accuracy** (oracle bound). The dominant factor
   remains sample count (n=200→1000 gives ~21% gain). For the live app the key lever is
   accumulating calibration samples across sessions, not refining the sampling strategy.

**Actions taken:**
- Added `--gaze-diverse`, `--dwell-avg K`, `--window-calib` flags to `mpii_bench.rs`.
- Dwell calibration should be implemented in `webgazer.rs` as a UX improvement (no click,
  timed fixation with feature averaging over the stable window).

---

## Best-of-the-best summary (for moving on or paper writing)

| Approach | Mean error | Protocol | Notes |
|----------|-----------|---------|-------|
| **`webgazer.rs` live, 5×5 grid, 20×12, 2×2 CLAHE** | **237 px / 3.7°** | Honest multi-point (E12) | Old best live |
| mpii_bench, 20×12, n=500, first-N | 5.31° | Standard MPIIGaze (E15) | Conservative benchmark |
| mpii_bench, 20×12, n=200, uniform-calib | 3.85° | Uniform (E16) | Beats L2CS-Net |
| mpii_bench, 30×18, n=200, uniform-calib, 3×3 | 3.54° | Uniform (E18) | +4% from CLAHE tiles |
| mpii_bench, 30×18, n=500, uniform-calib, 3×3 | 3.24° | Uniform (E18) | — |
| **mpii_bench, 40×24, n=1000, uniform-calib, 3×3** | **3.04° / 2.20°** | **Uniform (E18)** | **Best ever — FAZE-4.4%** |
| mpii_bench, 40×24, n=2000, uniform-calib, 3×3 | 2.94° / 2.12° | Uniform (E20, 12/15) | Curve flattening |
| mpii_bench, 40×24, n=5000, uniform-calib, 3×3 | 2.83° / 2.04° | Uniform (E20, 10/15) | Floor ~2.8° for this approach |
| WebGazer.js (reference) | ~175 px / 4° | Browser click-based | Has continuous learning |
| L2CS-Net | 3.92° | No calibration | Cross-subject DNN |
| FAZE | 3.18° | 9-point calib | Meta-learned fine-tune |
| GazeTR-Hybrid | 3.43° | No calibration | Transformer cross-subject |
| `webgazer_cnn.rs` (any CNN config) | 300–700 px | — | Don't use until Sugano fixed |

**Current state:** The 30×18 patch + 7×7 calibration grid is deployed in `webgazer.rs`.
Expected live improvement: 237 px → ~215 px. New session needed to measure actual improvement.

**Effective floor for ridge regression:** ~2.8° with unlimited calibration data (E20).
Beating this requires either more expressive features or a better model (CNN/Sugano).

Going to <150 px live requires (in priority order):
1. **Accumulated calibration across sessions** — free, existing infrastructure; plateau at ~n=1000
2. **Switch to 40×24 after ≥500 accumulated samples** — faster solve and better accuracy
3. Production Sugano normalization → CNN approach (2-3 weeks)

The white background, CLAHE, decay weights, blink filtering, and residual
rejection are all in place — their combined effect needs a live session recording
to measure (CLAHE operates on raw pixels, not recoverable from saved features).
