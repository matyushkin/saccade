//! Rust port of WebGazer.js — appearance-based webcam eye tracking via ridge regression.
//!
//! Pipeline:
//!   camera → rustface (face bbox) → PFLD landmarks → eye patches →
//!   10×6 grayscale histogram-equalized features (120-D) → ridge regression
//!   → Kalman filter → 4-point moving average → gaze cursor
//!
//! Calibration: 9 points shown in sequence. Look at the red dot and press SPACE
//! 5 times to capture samples (mimicking WebGazer's click-based calibration).
//!
//! Controls:
//!   SPACE — capture calibration sample (during calibration)
//!          | add bonus training sample at screen center (after calibration)
//!   C     — restart calibration
//!   ESC   — quit

use display_info::DisplayInfo;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use saccade::calib_state::{CalibrationState, EventResult, Phase};
use saccade::one_euro::OneEuroFilter2D;
use saccade::ridge::{self, RidgeRegressor, BOTH_EYES_FEAT_LEN};
use saccade::session::{CalibFrame, Session, ValidFrame};
use std::time::Instant;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const FACE_M: &str = "seeta_fd_frontal_v1.0.bin";
const FACE_U: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";
const PFLD_M: &str = "pfld.onnx";
const PFLD_U: &str = "https://github.com/cunjian/pytorch_face_landmark/raw/refs/heads/master/onnx/pfld.onnx";

fn dl(p: &str, u: &str) {
    if !std::path::Path::new(p).exists() {
        println!("Downloading {p}...");
        let _ = std::process::Command::new("curl").args(["-L", "-o", p, u]).status();
    }
}

fn load(p: &str) -> Model {
    tract_onnx::onnx().model_for_path(p).unwrap().into_optimized().unwrap().into_runnable().unwrap()
}

/// 9-point calibration grid: 8 perimeter points first, center LAST (WebGazer order).
fn calib_points(w: usize, h: usize) -> Vec<(f64, f64)> {
    let mx = w as f64 * 0.1;
    let my = h as f64 * 0.1;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    vec![
        (mx, my),                        // 1: top-left
        (cx, my),                        // 2: top-center
        (w as f64 - mx, my),             // 3: top-right
        (mx, cy),                        // 4: middle-left
        (w as f64 - mx, cy),             // 5: middle-right
        (mx, h as f64 - my),             // 6: bottom-left
        (cx, h as f64 - my),             // 7: bottom-center
        (w as f64 - mx, h as f64 - my),  // 8: bottom-right
        (cx, cy),                        // 9: center (LAST — like WebGazer)
    ]
}


fn main() {
    dl(FACE_M, FACE_U); dl(PFLD_M, PFLD_U);

    let mut face_det = rustface::create_detector(FACE_M).expect("face model");
    face_det.set_min_face_size(80);
    face_det.set_score_thresh(2.0);
    let pfld = load(PFLD_M);

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = loop {
        match Camera::new(CameraIndex::Index(0), format.clone()) {
            Ok(c) => break c,
            Err(e) => { eprintln!("Camera: {e}. Retry..."); std::thread::sleep(std::time::Duration::from_secs(2)); }
        }
    };
    camera.open_stream().expect("stream");

    let cam_res = camera.resolution();
    // Process at modest resolution for high FPS (rustface is the bottleneck)
    let sd = (cam_res.width() / 480).max(1) as usize;
    let cw = cam_res.width() as usize / sd;
    let ch = cam_res.height() as usize / sd;
    println!("Camera: {}x{} -> {}x{}", cam_res.width(), cam_res.height(), cw, ch);

    let displays = DisplayInfo::all().expect("displays");
    let primary = displays.iter().find(|d| d.is_primary).unwrap_or(&displays[0]);
    let sw = primary.width as usize;
    let sh = primary.height as usize;
    println!("Screen: {sw}x{sh}");

    // Bordered window — needed on macOS for keyboard focus
    let mut window = Window::new(
        "Saccade WebGazer Port [SPACE=sample, C=recalibrate, ESC=quit]",
        sw, sh,
        WindowOptions { ..WindowOptions::default() },
    ).unwrap();
    window.set_position(0, 0);
    window.set_target_fps(60);
    println!("Window opened. CLICK ON IT to give it focus, then press SPACE.");

    // Ridge regressor — STRONG regularization
    // Offline benchmark sweet spot: λ ∈ [1e6, 3e6] depending on session.
    // 120 features with only ~18 samples → severe overfit without strong λ.
    let mut ridge_reg = RidgeRegressor::new(
        50,
        3e6,                 // very strong ridge — best on recent benchmark
        BOTH_EYES_FEAT_LEN,
    );

    // Heavy smoothing: 16-point moving average + 1€ filter
    // (offline benchmark showed this is the sweet spot for our setup)
    let mut one_euro = OneEuroFilter2D::new(1.0, 0.05, 1.0);
    let mut gaze_history: Vec<(f64, f64)> = Vec::with_capacity(16);
    const SMOOTHING_WINDOW: usize = 16;

    let targets = calib_points(sw, sh);
    // Offline benchmark showed ~20 samples is optimal — use 2 per point × 9 = 18.
    // More clicks → user fatigue → less accurate → worse fit.
    const SAMPLES_PER_POINT: u32 = 2;
    let mut calib = CalibrationState::new(targets.len(), SAMPLES_PER_POINT);

    let mut buf = vec![0u32; sw * sh];
    let mut face_sm = SmRect::new(0.3);
    let mut no_face = 0u32;
    // Skip face detection on most frames (rustface ~30ms at 480x270)
    let mut frame_n = 0u64;

    // Validation state — collect gaze samples while user looks at center
    let mut validation_samples: Vec<(f64, f64)> = Vec::new();
    let mut validation_start_ms: u64 = 0;
    let validation_duration_ms: u64 = 3000;
    let mut validation_error_px: f64 = 0.0;
    let mut validation_error_std: f64 = 0.0;

    // Session recording for offline replay/benchmarking
    let mut session = Session::new(sw as u32, sh as u32, ((sw / 2) as f32, (sh / 2) as f32));
    let mut saved_session = false;

    let start = Instant::now();
    let mut fps_c = FpsC::new();

    // PIP preview
    let pip_w = 240usize;
    let pip_h = pip_w * ch / cw;

    println!("CLICK on each red dot {SAMPLES_PER_POINT} times to calibrate. Press ESC to quit.");

    // Mouse rising-edge detection
    let mut prev_mouse_down = false;
    let mut auto_started = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_ms = start.elapsed().as_millis() as u64;

        // Mouse click rising edge — more reliable than keyboard on macOS
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        let mouse_edge = mouse_down && !prev_mouse_down;
        prev_mouse_down = mouse_down;

        let mouse_pos = window.get_mouse_pos(MouseMode::Discard).unwrap_or((0.0, 0.0));

        let c_pressed = window.is_key_down(Key::C);
        if c_pressed {
            calib.restart();
            ridge_reg.clear();
            gaze_history.clear();
            one_euro.reset();
            println!("\nCalibration restarted.");
        }

        // Auto-start calibration when window opens (no keypress needed)
        if !auto_started && calib.phase() == Phase::Idle {
            calib.start();
            ridge_reg.clear();
            auto_started = true;
            println!("Calibration auto-started. Click on each red dot 5 times.");
        }

        // Capture frame
        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(i) => i, Err(_) => continue },
            Err(_) => continue,
        };
        let (fw, fh) = (decoded.width() as usize, decoded.height() as usize);
        let rf = decoded.as_raw();

        // Downscale
        let mut rgb = vec![0u8; cw * ch * 3];
        let mut gray = vec![0u8; cw * ch];
        for y in 0..ch { for x in 0..cw {
            let (sx, sy) = (x * sd, y * sd);
            if sx < fw && sy < fh {
                let si = (sy * fw + sx) * 3;
                let (r, g, b) = (rf[si], rf[si+1], rf[si+2]);
                let di = (y * cw + x) * 3;
                rgb[di] = r; rgb[di+1] = g; rgb[di+2] = b;
                gray[y*cw+x] = (0.299*r as f32 + 0.587*g as f32 + 0.114*b as f32) as u8;
            }
        }}

        // Face detection (skip every 3rd frame — heavy)
        frame_n += 1;
        let do_face = frame_n % 3 == 1 || !face_sm.init;
        let faces = if do_face {
            face_det.detect(&ImageData::new(&gray, cw as u32, ch as u32))
        } else {
            Vec::new()
        };
        let best = faces.iter().max_by_key(|f| { let b=f.bbox(); b.width()*b.height() });

        // Clear screen
        for px in buf.iter_mut() { *px = 0; }

        // PIP preview (top-right)
        let pip_x = sw - pip_w - 10;
        let pip_y = 10;
        for py in 0..pip_h { for px in 0..pip_w {
            let sx2 = px * cw / pip_w;
            let sy2 = py * ch / pip_h;
            if sx2 < cw && sy2 < ch {
                let si = (sy2 * cw + sx2) * 3;
                let r = rgb[si] as u32;
                let g = rgb[si+1] as u32;
                let b = rgb[si+2] as u32;
                let dx = pip_x + px;
                let dy = pip_y + py;
                if dx < sw && dy < sh { buf[dy * sw + dx] = (r<<16)|(g<<8)|b; }
            }
        }}
        draw_rect(&mut buf, sw, sh, pip_x.saturating_sub(1), pip_y.saturating_sub(1), pip_w+2, pip_h+2, 0xFFFFFF);

        // Update smoothed face from new detection if available
        if let Some(face) = best {
            no_face = 0;
            let bb = face.bbox();
            face_sm.update(bb.x().max(0) as f64, bb.y().max(0) as f64, bb.width() as f64, bb.height() as f64);
        } else if !do_face {
            // Skipped detection — no update, but keep using cached face
        } else {
            no_face += 1;
            if no_face > 60 { face_sm.init = false; no_face = 0; }
        }

        // Extract eye features (use cached face if any)
        let mut current_features: Option<Vec<f32>> = None;
        if face_sm.init {
            let (fx, fy, fwf, fhf) = face_sm.get();

            if let Some(lm) = run_pfld(&pfld, &gray, cw, ch, fx, fy, fwf, fhf) {
                // Draw landmarks in PIP
                for &(lx, ly) in &lm {
                    let px = pip_x + (lx as usize * pip_w / cw.max(1)).min(pip_w.saturating_sub(1));
                    let py = pip_y + (ly as usize * pip_h / ch.max(1)).min(pip_h.saturating_sub(1));
                    if px < sw && py < sh { buf[py * sw + px] = 0x00FF00; }
                }

                // Eye bounding boxes (iBUG 68: right eye 36-41, left eye 42-47)
                let right_patch = extract_eye_patch(&rgb, cw, ch, &lm, 36, 41);
                let left_patch = extract_eye_patch(&rgb, cw, ch, &lm, 42, 47);

                if let (Some((rp, rw, rh)), Some((lp, lw, lh))) = (right_patch, left_patch) {
                    let r_feat = ridge::extract_eye_features(&rp, rw, rh);
                    let l_feat = ridge::extract_eye_features(&lp, lw, lh);
                    let mut combined = Vec::with_capacity(BOTH_EYES_FEAT_LEN);
                    combined.extend_from_slice(&r_feat);
                    combined.extend_from_slice(&l_feat);
                    current_features = Some(combined);
                }
            }
        }

        // --- Idle screen: show visual prompt (only if auto-start failed) ---
        if false && calib.phase() == Phase::Idle {
            // Big centered text-like indicator: flashing bullseye in the middle
            let t = now_ms as f64 / 1000.0;
            let pulse = ((t * 2.0).sin() * 0.5 + 0.5);
            let cx = sw / 2;
            let cy = sh / 2;
            let r = (30.0 + pulse * 10.0) as usize;
            draw_filled_circle(&mut buf, sw, sh, cx, cy, r, 0x00AAFF);
            draw_ring(&mut buf, sw, sh, cx, cy, r + 15, 0xFFFFFF);
            draw_ring(&mut buf, sw, sh, cx, cy, r + 30, 0xFFFFFF);
            // Label area (just a differently-colored strip above)
            for y in (cy - 80)..(cy - 60) {
                for x in (cx.saturating_sub(300))..(cx + 300).min(sw) {
                    if y < sh { buf[y*sw+x] = 0xFFFFFF; }
                }
            }
            // And a differently-colored strip below
            for y in (cy + 60)..(cy + 80) {
                for x in (cx.saturating_sub(300))..(cx + 300).min(sw) {
                    if y < sh { buf[y*sw+x] = 0xFFFFFF; }
                }
            }
        }

        // --- Calibration phase ---
        if calib.phase() == Phase::Calibrating {
            let calib_idx = calib.current_point();
            let samples_at_current_point = calib.samples_at_current();
            let (tx, ty) = targets[calib_idx];

            // Color saturates from light pink → bright red as user clicks more
            // 0 clicks: light gray-pink, 5 clicks: bright pure red
            let progress = samples_at_current_point as f32 / SAMPLES_PER_POINT as f32;
            let red = (180.0 + 75.0 * progress) as u32;
            let green = (140.0 * (1.0 - progress)) as u32;
            let blue = (140.0 * (1.0 - progress)) as u32;
            let target_color = (red << 16) | (green << 8) | blue;

            // Bigger size as progress grows too
            let r = 22 + (progress * 8.0) as usize;
            draw_filled_circle(&mut buf, sw, sh, tx as usize, ty as usize, r, target_color);
            draw_ring(&mut buf, sw, sh, tx as usize, ty as usize, r + 3, 0xFFFFFF);
            draw_ring(&mut buf, sw, sh, tx as usize, ty as usize, r + 4, 0xFFFFFF);

            // Capture on mouse click anywhere on/near the target dot
            // (large hit radius — user just needs to click the area while looking at it)
            let click_dist = ((mouse_pos.0 as f64 - tx).powi(2) + (mouse_pos.1 as f64 - ty).powi(2)).sqrt();
            let click_on_target = click_dist < 80.0;
            if mouse_edge && click_on_target {
                let has_features = current_features.is_some();
                let result = calib.handle_capture(has_features);
                match result {
                    EventResult::SampleCaptured { point, sample } => {
                        if let Some(feats) = &current_features {
                            ridge_reg.add_sample(feats.clone(), tx as f32, ty as f32);
                            session.calibration.push(CalibFrame {
                                features: feats.clone(),
                                target_x: tx as f32,
                                target_y: ty as f32,
                            });
                        }
                        println!("  Point {}/{} sample {}/{}", point + 1, targets.len(), sample, SAMPLES_PER_POINT);
                    }
                    EventResult::NextPoint { point } => {
                        if let Some(feats) = &current_features {
                            ridge_reg.add_sample(feats.clone(), tx as f32, ty as f32);
                            session.calibration.push(CalibFrame {
                                features: feats.clone(),
                                target_x: tx as f32,
                                target_y: ty as f32,
                            });
                        }
                        println!("  Point {} complete, next: {}/{}", calib_idx + 1, point + 1, targets.len());
                    }
                    EventResult::CalibrationComplete => {
                        if let Some(feats) = &current_features {
                            ridge_reg.add_sample(feats.clone(), tx as f32, ty as f32);
                            session.calibration.push(CalibFrame {
                                features: feats.clone(),
                                target_x: tx as f32,
                                target_y: ty as f32,
                            });
                        }
                        println!("\nCalibration done: {} samples. Validating accuracy...", ridge_reg.sample_count());
                        validation_samples.clear();
                        gaze_history.clear();
                        one_euro.reset();
                        validation_start_ms = now_ms;
                    }
                    EventResult::Rejected => {
                        println!("  Capture rejected (no face detected — move into camera view)");
                    }
                    EventResult::Restarted => {}
                }
            } else if mouse_edge {
                println!("  Click was {:.0}px away from target — click closer to the red dot", click_dist);
            }
        }

        // --- Validation phase: user looks at center, we measure error ---
        if calib.phase() == Phase::Validating {
            let cx = sw / 2;
            let cy = sh / 2;

            // Big stable target — user just stares at this
            draw_filled_circle(&mut buf, sw, sh, cx, cy, 24, 0xFFFFFF);
            draw_ring(&mut buf, sw, sh, cx, cy, 30, 0x00AAFF);
            draw_ring(&mut buf, sw, sh, cx, cy, 31, 0x00AAFF);
            draw_filled_circle(&mut buf, sw, sh, cx, cy, 5, 0xFF0000);

            // Predict and collect gaze (with same smoothing as Running phase)
            if let Some(feats) = &current_features {
                if let Some((raw_x, raw_y)) = ridge_reg.predict(feats) {
                    let raw_gx = (raw_x as f64).clamp(0.0, sw as f64 - 1.0);
                    let raw_gy = (raw_y as f64).clamp(0.0, sh as f64 - 1.0);

                    // Apply same 16-point moving average + 1€ as Running phase
                    gaze_history.push((raw_gx, raw_gy));
                    if gaze_history.len() > SMOOTHING_WINDOW { gaze_history.remove(0); }
                    let n = gaze_history.len() as f64;
                    let avg_x = gaze_history.iter().map(|p| p.0).sum::<f64>() / n;
                    let avg_y = gaze_history.iter().map(|p| p.1).sum::<f64>() / n;
                    let t_sec = now_ms as f64 / 1000.0;
                    let (sm_x, sm_y) = one_euro.filter((avg_x, avg_y), t_sec);

                    // Skip first 1000ms (smoothing window fill + settling)
                    if now_ms.saturating_sub(validation_start_ms) > 1000 {
                        validation_samples.push((sm_x, sm_y));
                        // Also record into session for offline replay
                        session.validation.push(ValidFrame { features: feats.clone() });
                    }
                    // Show live gaze cursor (smoothed)
                    draw_filled_circle(&mut buf, sw, sh, sm_x as usize, sm_y as usize, 6, 0x00FF00);
                }
            }

            // Progress bar
            let elapsed = now_ms.saturating_sub(validation_start_ms);
            let progress = (elapsed as f64 / validation_duration_ms as f64).min(1.0);
            let bar_w = (sw as f64 * 0.5) as usize;
            let bar_x = (sw - bar_w) / 2;
            let bar_y = sh - 60;
            for x in 0..bar_w { for y in 0..10 { if bar_y+y < sh { buf[(bar_y+y)*sw+bar_x+x] = 0x222222; }}}
            let filled = (bar_w as f64 * progress) as usize;
            for x in 0..filled { for y in 0..10 { if bar_y+y < sh { buf[(bar_y+y)*sw+bar_x+x] = 0x00AAFF; }}}

            // Finish validation
            if elapsed >= validation_duration_ms {
                if validation_samples.len() >= 5 {
                    let n = validation_samples.len() as f64;
                    let mean_x = validation_samples.iter().map(|p| p.0).sum::<f64>() / n;
                    let mean_y = validation_samples.iter().map(|p| p.1).sum::<f64>() / n;
                    let target = (cx as f64, cy as f64);
                    let mean_err = ((mean_x - target.0).powi(2) + (mean_y - target.1).powi(2)).sqrt();
                    let var: f64 = validation_samples.iter()
                        .map(|p| (p.0 - mean_x).powi(2) + (p.1 - mean_y).powi(2))
                        .sum::<f64>() / n;
                    let std_dev = var.sqrt();
                    validation_error_px = mean_err;
                    validation_error_std = std_dev;
                    println!("\n=== Validation result ===");
                    println!("  Samples: {}", validation_samples.len());
                    println!("  Mean error: {mean_err:.0} px");
                    println!("  Precision (std): {std_dev:.0} px");
                    println!("  Press C to recalibrate, ESC to quit.");
                }

                // Save session for offline replay/benchmarking (once per session)
                if !saved_session {
                    let path = "saccade_session.bin";
                    match session.save(path) {
                        Ok(_) => println!("Session saved: {} ({} calib + {} validation samples)",
                            path, session.calibration.len(), session.validation.len()),
                        Err(e) => eprintln!("Failed to save session: {e}"),
                    }
                    saved_session = true;
                }

                calib.finish_validation();
            }
        }

        // --- Running phase: predict gaze and show cursor ---
        if calib.phase() == Phase::Running {
            if let Some(feats) = &current_features {
                if let Some((raw_x, raw_y)) = ridge_reg.predict(feats) {
                    let clamped_x = (raw_x as f64).clamp(0.0, sw as f64 - 1.0);
                    let clamped_y = (raw_y as f64).clamp(0.0, sh as f64 - 1.0);

                    // 16-point moving average (sweet spot from offline benchmark)
                    gaze_history.push((clamped_x, clamped_y));
                    if gaze_history.len() > SMOOTHING_WINDOW { gaze_history.remove(0); }
                    let n = gaze_history.len() as f64;
                    let avg_x = gaze_history.iter().map(|p| p.0).sum::<f64>() / n;
                    let avg_y = gaze_history.iter().map(|p| p.1).sum::<f64>() / n;

                    // 1€ filter on top of moving average for jitter cleanup
                    let t_sec = now_ms as f64 / 1000.0;
                    let (gx_f, gy_f) = one_euro.filter((avg_x, avg_y), t_sec);

                    let gx = (gx_f as usize).min(sw - 1);
                    let gy = (gy_f as usize).min(sh - 1);

                    // Draw gaze cursor
                    draw_filled_circle(&mut buf, sw, sh, gx, gy, 10, 0x00FF00);
                    draw_ring(&mut buf, sw, sh, gx, gy, 20, 0xFFFFFF);
                    draw_ring(&mut buf, sw, sh, gx, gy, 30, 0x00FF00);
                }
            }

            // Show validation results in corner
            if validation_error_px > 0.0 {
                let bar_y = 50;
                let bar_x = 20;
                // Draw a small status bar showing accuracy
                let err_color = if validation_error_px < 100.0 { 0x00FF00 }
                                else if validation_error_px < 200.0 { 0xFFFF00 }
                                else { 0xFF0000 };
                for y in 0..30 { for x in 0..200 { if bar_y+y < sh && bar_x+x < sw { buf[(bar_y+y)*sw+bar_x+x] = 0x222222; }}}
                let bar_fill = (200.0 - validation_error_px.min(200.0)) as usize;
                for y in 0..30 { for x in 0..bar_fill { if bar_y+y < sh && bar_x+x < sw { buf[(bar_y+y)*sw+bar_x+x] = err_color; }}}
            }
        }

        // FPS / status bar
        fps_c.tick();
        let f = fps_c.fps();
        let bar = (f as usize * 5).min(sw);
        let bc = if f > 20.0 { 0x00FF00 } else if f > 10.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar { for y in 0..4 { buf[y*sw+x] = bc; } }

        window.update_with_buffer(&buf, sw, sh).unwrap();

        if fps_c.count % 30 == 0 {
            let mode = match calib.phase() {
                Phase::Idle => "IDLE",
                Phase::Calibrating => "CALIB",
                Phase::Validating => "VALID",
                Phase::Running => "RUN",
            };
            print!("\r[{mode}] FPS:{f:.0} | Samples:{} | Point:{}/{}    ",
                ridge_reg.sample_count(), calib.current_point().min(targets.len()), targets.len());
        }
    }
    println!("\nDone.");
}

/// Extract an eye RGB patch using PFLD landmarks (iBUG 68 eye indices).
fn extract_eye_patch(
    rgb: &[u8], w: usize, h: usize,
    lm: &[(f32, f32)], si: usize, ei: usize,
) -> Option<(Vec<u8>, usize, usize)> {
    let pts = &lm[si..=ei];
    let min_x = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let max_x = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let min_y = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let max_y = pts.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    // Add 30% horizontal, 80% vertical margin
    let mx = (max_x - min_x) * 0.3;
    let my = (max_y - min_y) * 0.8;
    let rx = (min_x - mx).max(0.0) as usize;
    let ry = (min_y - my).max(0.0) as usize;
    let rw = ((max_x - min_x) + 2.0*mx) as usize;
    let rh = ((max_y - min_y) + 2.0*my) as usize;
    if rx+rw > w || ry+rh > h || rw < 8 || rh < 6 { return None; }

    let mut patch = vec![0u8; rw * rh * 3];
    for y in 0..rh {
        for x in 0..rw {
            let si2 = ((ry+y) * w + (rx+x)) * 3;
            let di = (y * rw + x) * 3;
            patch[di]   = rgb[si2];
            patch[di+1] = rgb[si2+1];
            patch[di+2] = rgb[si2+2];
        }
    }
    Some((patch, rw, rh))
}

fn run_pfld(m: &Model, g: &[u8], w: usize, h: usize, fx: usize, fy: usize, fw: usize, fh: usize) -> Option<Vec<(f32,f32)>> {
    if fx+fw>w||fy+fh>h||fw<20||fh<20 { return None; }
    let s=112; let mut d=vec![0.0f32;3*s*s];
    for y in 0..s{for x in 0..s{let(sx,sy)=(fx+x*fw/s,fy+y*fh/s);let v=if sx<w&&sy<h{g[sy*w+sx]as f32/255.0}else{0.0};d[y*s+x]=v;d[s*s+y*s+x]=v;d[2*s*s+y*s+x]=v;}}
    let t = Tensor::from(tract_ndarray::Array4::from_shape_vec((1,3,s,s),d).ok()?).into();
    let r = m.run(tvec![t]).ok()?;
    let out = r[0].to_array_view::<f32>().ok()?;
    let flat = out.as_slice()?;
    if flat.len()<136{return None;}
    Some((0..68).map(|i|(flat[i*2]*fw as f32+fx as f32, flat[i*2+1]*fh as f32+fy as f32)).collect())
}

// --- UI helpers ---
struct SmRect{x:f64,y:f64,w:f64,h:f64,a:f64,init:bool}
impl SmRect{fn new(a:f64)->Self{Self{x:0.0,y:0.0,w:0.0,h:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64,w:f64,h:f64){if!self.init{self.x=x;self.y=y;self.w=w;self.h=h;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;self.w=a*w+(1.0-a)*self.w;self.h=a*h+(1.0-a)*self.h;}}fn get(&self)->(usize,usize,usize,usize){(self.x.round()as usize,self.y.round()as usize,self.w.round().max(1.0)as usize,self.h.round().max(1.0)as usize)}}
fn draw_filled_circle(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for dy in 0..=r{for dx in 0..=r{if dx*dx+dy*dy<=r*r{for&(sx,sy)in&[(cx+dx,cy+dy),(cx.wrapping_sub(dx),cy+dy),(cx+dx,cy.wrapping_sub(dy)),(cx.wrapping_sub(dx),cy.wrapping_sub(dy))]{if sx<w&&sy<h{b[sy*w+sx]=c;}}}}}}
fn draw_ring(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for i in 0..64{let t=2.0*std::f64::consts::PI*i as f64/64.0;let x=(cx as f64+r as f64*t.cos()).round()as i32;let y=(cy as f64+r as f64*t.sin()).round()as i32;if x>=0&&(x as usize)<w&&y>=0&&(y as usize)<h{b[y as usize*w+x as usize]=c;}}}
fn draw_rect(b:&mut[u32],w:usize,h:usize,x:usize,y:usize,rw:usize,rh:usize,c:u32){for dx in 0..rw{let px=x+dx;if px<w{if y<h{b[y*w+px]=c;}let by=y+rh.saturating_sub(1);if by<h{b[by*w+px]=c;}}}for dy in 0..rh{let py=y+dy;if py<h{if x<w{b[py*w+x]=c;}let bx=x+rw.saturating_sub(1);if bx<w{b[py*w+bx]=c;}}}}
struct FpsC{t:Instant,count:u64,fps:f64}
impl FpsC{fn new()->Self{Self{t:Instant::now(),count:0,fps:0.0}}fn tick(&mut self){self.count+=1;let e=self.t.elapsed().as_secs_f64();if e>=1.0{self.fps=self.count as f64/e;self.count=0;self.t=Instant::now();}}fn fps(&self)->f64{self.fps}}
