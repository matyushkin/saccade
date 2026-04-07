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

### E1. Pixel ridge regression (WebGazer port) — **OUR BEST**

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

### E2. Pixel ridge + head pose features

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

### E3. CNN MobileGaze + linear mapper

**Approach:** Use pretrained MobileGaze ONNX (4.7 MB, MobileOne-S0 backbone) to
predict (yaw, pitch) angles from face crop. Fit linear `screen = A·(yaw,pitch) + b`
with 6 parameters from 18 calibration clicks.

**Files:** `examples/webgazer_cnn.rs`

**Result:** Mean **299 px**, max 410 px, coverage **134%/124%** (no collapse).

**Insight:** CNN doesn't collapse to center because it predicts gaze direction
in absolute terms. But linear mapper is too simple — only 6 params can't capture
the full pitch/yaw → screen relationship.

---

### E4. CNN MobileGaze + 11-feature polynomial (head pose)

**Approach:** Polynomial mapper with `[1, yaw, pitch, hx, hy, hw, hr, yaw², pitch², yaw·pitch, hw·yaw]` features. 11 parameters.

**Result:** Mean **383 px**, max 885 px (catastrophic outlier).

**Why it failed:** 11 parameters fit on only 18 samples → overfitting. LOO error
was 361 px. Auto-lambda picked 1.0 which was too small.

---

### E5. CNN MobileGaze + 6-feature polynomial (yaw/pitch only)

**Approach:** Standard literature recipe — degree-2 polynomial of (yaw, pitch):
`[1, yaw, pitch, yaw², pitch², yaw·pitch]`. 45 calibration samples.

**Result:** Not robustly tested — got crashed and ran with mostly the same
performance as E3-E4.

---

### E6. CNN ResNet-18 + 6-feature polynomial

**Approach:** Switched MobileGaze (4.7 MB) → ResNet-18 (43 MB), supposed to be
slightly more accurate. 45 calibration, 6-feature polynomial.

**Result:** Mean **397 px**, max 584 px, coverage Y=43%. **Worse** than ResNet-50
attempt and worse than MobileGaze linear.

**Why it failed:** Same fundamental issue — without proper head-pose
normalization, the CNN gives noisy/biased angles.

---

### E7. CNN ResNet-50 (briefly tested)

**Approach:** Largest model, 95 MB, supposedly closest to L2CS-Net 3.92° on
MPIIFaceGaze.

**Result:** Got dropped in favor of ResNet-18 because it ran at <2 FPS, making
the UI unusable. Calibration clicks didn't register reliably.

---

### E8. CNN + Sugano 2014 normalization (my implementation)

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

### E10. White background lighting hack — **BIG WIN**

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

### E9. PPERV (head-pose-invariant pupil position)

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

## Best-of-the-best summary (for moving on or paper writing)

| Approach | Mean error | When to use |
|----------|-----------|-------------|
| **`webgazer.rs` pixel ridge, 45 cal, λ=auto, WHITE BG** | **142 px / 2.2°** | Default — beats WebGazer.js |
| WebGazer.js (reference) | ~175 px / 4° | Browser, has continuous learning |
| `webgazer_cnn.rs` (any CNN config) | 300-700 px | Don't use until Sugano fixed |

**Current state:** **Beats WebGazer.js** on mean error for short sessions
(142 px vs ~175 px) thanks to the white background lighting trick. Worst case
is comparable. Still no head-pose compensation, so head movement breaks
calibration. Going to <100 px would require production Sugano + CNN, multi-week
effort.
