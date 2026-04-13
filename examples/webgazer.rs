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
use saccade::ridge::{self, RidgeRegressor, RidgeSample, BOTH_EYES_FEAT_LEN};

const HEAD_POSE_FEAT_LEN: usize = 6;
const TOTAL_FEAT_LEN: usize = BOTH_EYES_FEAT_LEN + HEAD_POSE_FEAT_LEN; // 126
const EAR_BLINK_THRESHOLD: f32 = 0.15;
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

/// Smooth pursuit trajectory: horizontal meander (4 sweeps) + vertical meander (4 sweeps).
/// Total duration: 18 seconds, ~150-200 samples at 10 FPS.
/// Returns target position at time `t` seconds.
fn pursuit_target(t: f64, sw: f64, sh: f64) -> (f64, f64) {
    let mx = sw * 0.1;
    let my = sh * 0.1;
    let usable_w = sw - 2.0 * mx;
    let usable_h = sh - 2.0 * my;

    // 0-9 sec: horizontal meander (4 horizontal sweeps stepping down)
    // 9-18 sec: vertical meander (4 vertical sweeps stepping right)
    if t < 9.0 {
        let phase = t / 9.0; // 0..1
        let sweep_progress = phase * 4.0; // 0..4
        let sweep_idx = sweep_progress.floor() as usize;
        let local = sweep_progress - sweep_idx as f64; // 0..1 within sweep

        let y = my + usable_h * (sweep_idx as f64 + 0.5) / 4.0;
        // Even sweeps go left→right, odd sweeps go right→left
        let x = if sweep_idx % 2 == 0 {
            mx + usable_w * local
        } else {
            mx + usable_w * (1.0 - local)
        };
        (x, y)
    } else {
        let phase = (t - 9.0) / 9.0;
        let sweep_progress = phase * 4.0;
        let sweep_idx = sweep_progress.floor() as usize;
        let local = sweep_progress - sweep_idx as f64;

        let x = mx + usable_w * (sweep_idx as f64 + 0.5) / 4.0;
        let y = if sweep_idx % 2 == 0 {
            my + usable_h * local
        } else {
            my + usable_h * (1.0 - local)
        };
        (x, y)
    }
}

/// 5x5 grid calibration: 25 cells with random number labels.
/// User clicks numbers 1..25 in order, each click adds a sample.
fn grid_5x5_centers(w: usize, h: usize) -> Vec<(f64, f64)> {
    let mx = w as f64 * 0.08;
    let my = h as f64 * 0.08;
    let usable_w = w as f64 - 2.0 * mx;
    let usable_h = h as f64 - 2.0 * my;
    let mut pts = Vec::with_capacity(25);
    for row in 0..5 {
        for col in 0..5 {
            // Cell center
            let cx = mx + usable_w * (col as f64 + 0.5) / 5.0;
            let cy = my + usable_h * (row as f64 + 0.5) / 5.0;
            pts.push((cx, cy));
        }
    }
    pts
}

/// Build a random permutation of 1..25 → maps cell index → number to display.
fn random_number_map() -> [u32; 25] {
    let mut nums: [u32; 25] = std::array::from_fn(|i| (i + 1) as u32);
    // Fisher-Yates shuffle using time-based seed
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(12345);
    let mut state = seed;
    for i in (1..25).rev() {
        // xorshift
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        nums.swap(i, j);
    }
    nums
}

/// Draw a digit (0-9) as a 7-segment style block at (cx, cy).
/// Each segment is a thick line. Total digit size ~30×50 px.
fn draw_digit(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, digit: u32, color: u32) {
    // 7-segment layout:
    //  aaa
    // f   b
    // f   b
    //  ggg
    // e   c
    // e   c
    //  ddd
    let segments: [bool; 7] = match digit {
        0 => [true, true, true, true, true, true, false],
        1 => [false, true, true, false, false, false, false],
        2 => [true, true, false, true, true, false, true],
        3 => [true, true, true, true, false, false, true],
        4 => [false, true, true, false, false, true, true],
        5 => [true, false, true, true, false, true, true],
        6 => [true, false, true, true, true, true, true],
        7 => [true, true, true, false, false, false, false],
        8 => [true, true, true, true, true, true, true],
        9 => [true, true, true, true, false, true, true],
        _ => [false; 7],
    };
    let half_w = 14i32;
    let half_h = 22i32;
    let thick = 4i32;

    let draw_h_seg = |buf: &mut [u32], y_offset: i32| {
        for dy in -thick/2..=thick/2 {
            for dx in (-half_w + 4)..(half_w - 3) {
                let x = cx as i32 + dx;
                let y = cy as i32 + y_offset + dy;
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    buf[y as usize * w + x as usize] = color;
                }
            }
        }
    };
    let draw_v_seg = |buf: &mut [u32], x_offset: i32, y_start: i32, y_end: i32| {
        for dy in y_start..=y_end {
            for dx in -thick/2..=thick/2 {
                let x = cx as i32 + x_offset + dx;
                let y = cy as i32 + dy;
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    buf[y as usize * w + x as usize] = color;
                }
            }
        }
    };

    // a: top horizontal
    if segments[0] { draw_h_seg(buf, -half_h); }
    // b: top right vertical
    if segments[1] { draw_v_seg(buf, half_w - 4, -half_h + 3, -2); }
    // c: bottom right vertical
    if segments[2] { draw_v_seg(buf, half_w - 4, 2, half_h - 3); }
    // d: bottom horizontal
    if segments[3] { draw_h_seg(buf, half_h); }
    // e: bottom left vertical
    if segments[4] { draw_v_seg(buf, -half_w + 4, 2, half_h - 3); }
    // f: top left vertical
    if segments[5] { draw_v_seg(buf, -half_w + 4, -half_h + 3, -2); }
    // g: middle horizontal
    if segments[6] { draw_h_seg(buf, 0); }
}

/// Draw a number 1-25 (one or two digits) centered at (cx, cy).
fn draw_number(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, n: u32, color: u32) {
    if n < 10 {
        draw_digit(buf, w, h, cx, cy, n, color);
    } else {
        let tens = n / 10;
        let ones = n % 10;
        draw_digit(buf, w, h, cx.saturating_sub(20), cy, tens, color);
        draw_digit(buf, w, h, cx + 20, cy, ones, color);
    }
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

    // Warm up camera (let AE settle) then lock exposure to prevent drift
    // between calibration and gaze prediction.
    {
        use nokhwa::utils::{ControlValueSetter, KnownCameraControl};
        for _ in 0..10 {
            let _ = camera.frame();
        }
        match camera.camera_control(KnownCameraControl::Exposure) {
            Ok(ctrl) => {
                let setter = ctrl.value();
                match camera.set_camera_control(KnownCameraControl::Exposure, setter) {
                    Ok(_) => println!("Camera exposure locked."),
                    Err(e) => println!("Note: couldn't lock exposure ({e}), continuing."),
                }
            }
            Err(e) => println!("Note: exposure control not available ({e})."),
        }
    }

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

    // Ridge regressor — eye features + 6 head pose features = 126-D
    let mut ridge_reg = RidgeRegressor::new(
        200,                 // larger buffer — accumulate user clicks during use
        1e4,                 // initial guess; auto-tuned via LOO CV after calibration
        TOTAL_FEAT_LEN,
    );

    // Heavy smoothing: 16-point moving average + 1€ filter
    // (offline benchmark showed this is the sweet spot for our setup)
    let mut one_euro = OneEuroFilter2D::new(1.0, 0.05, 1.0);
    let mut gaze_history: Vec<(f64, f64)> = Vec::with_capacity(16);
    const SMOOTHING_WINDOW: usize = 16;

    // 5x5 grid game calibration: 25 cells with random number labels, played 2 rounds.
    // Round 1: numbers 1..25 in random positions. Round 2: re-shuffle and play again.
    // Total: 50 samples across 25 unique screen positions.
    let grid_centers = grid_5x5_centers(sw, sh);
    let mut number_map = random_number_map();
    let mut next_number: u32 = 1;
    let mut grid_calib_done = false;
    let mut grid_round: u32 = 1;
    const GRID_ROUNDS: u32 = 2;

    // Old 9-point click flow disabled in favor of 5x5 game
    let targets: Vec<(f64, f64)> = grid_centers.clone();
    const SAMPLES_PER_POINT: u32 = 1;
    let mut calib = CalibrationState::new(targets.len(), SAMPLES_PER_POINT);

    let mut buf = vec![0u32; sw * sh];
    let mut face_sm = SmRect::new(0.3);
    let mut no_face = 0u32;
    // Skip face detection on most frames (rustface ~30ms at 480x270)
    let mut frame_n = 0u64;

    // Smooth pursuit phase — disabled by default. Made things worse (E11 in EXPERIMENTS.md).
    let mut pursuit_active = false;
    let mut pursuit_start_ms: u64 = 0;
    let pursuit_duration_ms: u64 = 18_000;

    // Multi-point validation: 5 different targets, click each one,
    // we measure error vs prediction for each
    let validation_targets: Vec<(f64, f64)> = vec![
        (sw as f64 * 0.2, sh as f64 * 0.2),  // top-left interior
        (sw as f64 * 0.8, sh as f64 * 0.2),  // top-right interior
        (sw as f64 / 2.0, sh as f64 / 2.0),  // center
        (sw as f64 * 0.2, sh as f64 * 0.8),  // bottom-left interior
        (sw as f64 * 0.8, sh as f64 * 0.8),  // bottom-right interior
    ];
    let mut validation_idx = 0usize;
    // Per-validation-point: collected (predicted, target) pairs
    let mut validation_results: Vec<(f64, f64, f64, f64)> = Vec::new(); // (px, py, tx, ty)
    let mut prev_val_mouse_down = false;

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
        for px in buf.iter_mut() { *px = 0xFFFFFF; }

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

                    // Head pose features (6) — scaled to similar magnitude as eye pixel features (~100)
                    // face_x, face_y normalized to [0, 100] across image
                    let head_x = (fx as f32 + fwf as f32 / 2.0) / cw as f32 * 100.0;
                    let head_y = (fy as f32 + fhf as f32 / 2.0) / ch as f32 * 100.0;
                    let head_w = fwf as f32 / cw as f32 * 100.0;
                    let head_h = fhf as f32 / ch as f32 * 100.0;
                    // Head roll from eye corners (right outer 36 to left outer 45)
                    let dx = lm[45].0 - lm[36].0;
                    let dy = lm[45].1 - lm[36].1;
                    let head_roll = dy.atan2(dx) * 100.0; // radians × 100 ≈ pixel scale
                    // Inter-eye distance (proxy for camera distance)
                    let r_eye_cx = (lm[36].0 + lm[39].0) / 2.0;
                    let l_eye_cx = (lm[42].0 + lm[45].0) / 2.0;
                    let r_eye_cy = (lm[36].1 + lm[39].1) / 2.0;
                    let l_eye_cy = (lm[42].1 + lm[45].1) / 2.0;
                    let inter_eye = ((l_eye_cx - r_eye_cx).powi(2) + (l_eye_cy - r_eye_cy).powi(2)).sqrt();

                    let mut combined = Vec::with_capacity(TOTAL_FEAT_LEN);
                    combined.extend_from_slice(&r_feat);
                    combined.extend_from_slice(&l_feat);
                    combined.push(head_x);
                    combined.push(head_y);
                    combined.push(head_w);
                    combined.push(head_h);
                    combined.push(head_roll);
                    combined.push(inter_eye);
                    let ear_r = eye_aspect_ratio(&lm, 36);
                    let ear_l = eye_aspect_ratio(&lm, 42);
                    if ear_r.min(ear_l) < EAR_BLINK_THRESHOLD {
                        // Eye closed/blinking — skip this frame
                    } else {
                        current_features = Some(combined);
                    }
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

        // --- Calibration phase: 5x5 number grid game ---
        if calib.phase() == Phase::Calibrating && !grid_calib_done {
            let mx = sw as f64 * 0.08;
            let my = sh as f64 * 0.08;
            let usable_w = sw as f64 - 2.0 * mx;
            let usable_h = sh as f64 - 2.0 * my;
            let cell_w = usable_w / 5.0;
            let cell_h = usable_h / 5.0;

            // Draw all cell borders + numbers (numbers fade as user clicks them)
            for cell_idx in 0..25 {
                let row = cell_idx / 5;
                let col = cell_idx % 5;
                let cell_x = (mx + cell_w * col as f64) as usize;
                let cell_y = (my + cell_h * row as f64) as usize;
                let cw_px = cell_w as usize;
                let ch_px = cell_h as usize;

                // Cell border (dark gray on white)
                draw_rect(&mut buf, sw, sh, cell_x, cell_y, cw_px, ch_px, 0xCCCCCC);

                let n = number_map[cell_idx];
                let color = if n < next_number {
                    0xCCCCCC // already clicked, faded
                } else if n == next_number {
                    0xCC0000 // current target — red, prominent
                } else {
                    0x444444 // dark gray
                };

                let cx = cell_x + cw_px / 2;
                let cy = cell_y + ch_px / 2;

                // Highlight current target cell with light background
                if n == next_number {
                    for y in 0..ch_px {
                        for x in 0..cw_px {
                            if cell_x + x < sw && cell_y + y < sh {
                                buf[(cell_y + y) * sw + cell_x + x] = 0xFFEEEE;
                            }
                        }
                    }
                    // Re-draw border
                    draw_rect(&mut buf, sw, sh, cell_x, cell_y, cw_px, ch_px, 0xCC0000);
                }

                draw_number(&mut buf, sw, sh, cx, cy, n, color);
            }

            // Instructions at top
            // Just leave them in stdout — drawing text is hard
            // (The user can see "Find: N" with the highlighted cell)

            // Click handling
            if mouse_edge {
                let mx_click = mouse_pos.0 as f64;
                let my_click = mouse_pos.1 as f64;
                // Find which cell was clicked
                let col = ((mx_click - mx) / cell_w).floor() as i32;
                let row = ((my_click - my) / cell_h).floor() as i32;
                if col >= 0 && col < 5 && row >= 0 && row < 5 {
                    let clicked_idx = (row * 5 + col) as usize;
                    let clicked_n = number_map[clicked_idx];
                    if clicked_n == next_number {
                        // Correct! Capture sample at cell center
                        let cell_cx = mx + cell_w * col as f64 + cell_w / 2.0;
                        let cell_cy = my + cell_h * row as f64 + cell_h / 2.0;
                        if let Some(feats) = &current_features {
                            ridge_reg.add_sample(feats.clone(), cell_cx as f32, cell_cy as f32);
                            session.calibration.push(CalibFrame {
                                features: feats.clone(),
                                target_x: cell_cx as f32,
                                target_y: cell_cy as f32,
                            });
                            println!("  Found {next_number}/25 at cell ({col},{row})");
                        }
                        next_number += 1;
                        if next_number > 25 {
                            if grid_round < GRID_ROUNDS {
                                // Start next round with reshuffled numbers
                                grid_round += 1;
                                next_number = 1;
                                number_map = random_number_map();
                                println!("\nRound {grid_round}/{GRID_ROUNDS} starting...");
                            } else {
                                grid_calib_done = true;
                                // Move state machine to validation phase
                                while calib.phase() == Phase::Calibrating {
                                    let _ = calib.handle_capture(true);
                                }
                                // Auto-tune lambda
                                let candidates = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6];
                                if let Some(best_lam) = ridge_reg.auto_lambda(&candidates) {
                                    let loo_err = ridge_reg.loo_error(best_lam);
                                    ridge_reg.set_lambda(best_lam);
                                    println!("\nGrid calibration done ({} rounds): {} samples. Auto λ={best_lam:.0e} (LOO err: {loo_err:.0} px)",
                                        GRID_ROUNDS, ridge_reg.sample_count());
                                }
                                // Post-calibration residual filtering: remove samples where the model's
                                // training error is an outlier (mean + 2σ). Removes blinked/saccadic clicks.
                                {
                                    let _init_lam = ridge_reg.lambda;
                                    if ridge_reg.sample_count() >= 6 {
                                        let mut residuals: Vec<f64> = Vec::new();
                                        let all_feats: Vec<Vec<f32>> = ridge_reg.samples.iter().map(|s| s.features.clone()).collect();
                                        let all_tx: Vec<f32> = ridge_reg.samples.iter().map(|s| s.target_x).collect();
                                        let all_ty: Vec<f32> = ridge_reg.samples.iter().map(|s| s.target_y).collect();
                                        for (i, feats) in all_feats.iter().enumerate() {
                                            if let Some((px, py)) = ridge_reg.predict(feats) {
                                                let dx = px as f64 - all_tx[i] as f64;
                                                let dy = py as f64 - all_ty[i] as f64;
                                                residuals.push((dx*dx + dy*dy).sqrt());
                                            } else {
                                                residuals.push(0.0);
                                            }
                                        }
                                        let mean_r: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
                                        let var_r: f64 = residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / residuals.len() as f64;
                                        let std_r = var_r.sqrt();
                                        let threshold = mean_r + 2.0 * std_r;
                                        let before = ridge_reg.sample_count();
                                        let good: Vec<RidgeSample> = ridge_reg.samples.iter().zip(residuals.iter())
                                            .filter(|(_, &r)| r <= threshold)
                                            .map(|(s, _)| s.clone())
                                            .collect();
                                        let removed = before - good.len();
                                        ridge_reg.clear();
                                        for s in good {
                                            ridge_reg.add_sample(s.features, s.target_x, s.target_y);
                                        }
                                        if removed > 0 {
                                            println!("Residual filtering: removed {removed} outlier samples (threshold {threshold:.0} px).");
                                            // Re-tune lambda on cleaned data
                                            let candidates2 = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6];
                                            if let Some(best_lam) = ridge_reg.auto_lambda(&candidates2) {
                                                ridge_reg.set_lambda(best_lam);
                                                println!("Re-tuned λ={best_lam:.0e} after filtering.");
                                            }
                                        }
                                    }
                                }
                                println!("Now click each blue dot for validation.");
                                validation_idx = 0;
                                validation_results.clear();
                                gaze_history.clear();
                                one_euro.reset();
                            }
                        }
                    } else {
                        println!("  Wrong cell (clicked {clicked_n}, expected {next_number})");
                    }
                }
            }
        }

        // --- Smooth pursuit phase: animated trajectory after click calibration ---
        if pursuit_active {
            let elapsed = now_ms.saturating_sub(pursuit_start_ms);
            let t = elapsed as f64 / 1000.0;
            let (ptx, pty) = pursuit_target(t, sw as f64, sh as f64);

            // Draw moving dot
            draw_filled_circle(&mut buf, sw, sh, ptx as usize, pty as usize, 16, 0xCC0000);
            draw_ring(&mut buf, sw, sh, ptx as usize, pty as usize, 22, 0x000000);
            draw_filled_circle(&mut buf, sw, sh, ptx as usize, pty as usize, 4, 0xFFFFFF);

            // Draw faint trail (last few seconds)
            for back in 1..20 {
                let prev_t = (t - back as f64 * 0.05).max(0.0);
                if prev_t > 0.0 {
                    let (px, py) = pursuit_target(prev_t, sw as f64, sh as f64);
                    let alpha = 1.0 - back as f64 / 20.0;
                    let gray = (200.0 - 50.0 * alpha) as u32;
                    let color = (gray << 16) | (gray << 8) | gray;
                    draw_filled_circle(&mut buf, sw, sh, px as usize, py as usize, 2, color);
                }
            }

            // Progress bar at bottom
            let progress = (elapsed as f64 / pursuit_duration_ms as f64).min(1.0);
            let bar_w = (sw as f64 * 0.5) as usize;
            let bar_x = (sw - bar_w) / 2;
            let bar_y = sh - 50;
            for x in 0..bar_w { for y in 0..10 { if bar_y+y < sh { buf[(bar_y+y)*sw+bar_x+x] = 0xCCCCCC; }}}
            let filled = (bar_w as f64 * progress) as usize;
            for x in 0..filled { for y in 0..10 { if bar_y+y < sh { buf[(bar_y+y)*sw+bar_x+x] = 0xCC0000; }}}

            // Collect sample every frame after a brief delay (let user catch up)
            // Smooth pursuit has ~100-150ms latency, so we offset the target by that
            // (predict where the user is actually looking now, not where the dot is now)
            let lag_ms = 150.0;
            let lagged_t = (t * 1000.0 - lag_ms).max(0.0) / 1000.0;
            let (sample_tx, sample_ty) = pursuit_target(lagged_t, sw as f64, sh as f64);

            if elapsed > 500 && elapsed < pursuit_duration_ms {
                if let Some(feats) = &current_features {
                    ridge_reg.add_sample(feats.clone(), sample_tx as f32, sample_ty as f32);
                    session.calibration.push(CalibFrame {
                        features: feats.clone(),
                        target_x: sample_tx as f32,
                        target_y: sample_ty as f32,
                    });
                }
            }

            // Finish pursuit
            if elapsed >= pursuit_duration_ms {
                pursuit_active = false;
                println!("\nPursuit done. Total samples: {}", ridge_reg.sample_count());
                // Now run auto-tune lambda on full enriched buffer
                let candidates = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6];
                if let Some(best_lam) = ridge_reg.auto_lambda(&candidates) {
                    let loo_err = ridge_reg.loo_error(best_lam);
                    ridge_reg.set_lambda(best_lam);
                    println!("Auto λ={best_lam:.0e} (LOO err: {loo_err:.0} px)");
                }
                println!("Now click each blue dot.");
                validation_idx = 0;
                validation_results.clear();
                gaze_history.clear();
                one_euro.reset();
            }
        }

        // --- Multi-point validation: user CLICKS each validation target ---
        if calib.phase() == Phase::Validating && validation_idx < validation_targets.len() {
            let (vtx, vty) = validation_targets[validation_idx];

            // Draw target on white bg — dark blue dot with black ring
            draw_filled_circle(&mut buf, sw, sh, vtx as usize, vty as usize, 24, 0x0066CC);
            draw_ring(&mut buf, sw, sh, vtx as usize, vty as usize, 30, 0x000000);
            draw_ring(&mut buf, sw, sh, vtx as usize, vty as usize, 31, 0x000000);
            draw_filled_circle(&mut buf, sw, sh, vtx as usize, vty as usize, 4, 0x000000);

            // Progress dots for all validation positions
            for (i, &(px, py)) in validation_targets.iter().enumerate() {
                let c = if i < validation_idx { 0x009900 }
                        else if i == validation_idx { 0x0066CC }
                        else { 0xAAAAAA };
                draw_filled_circle(&mut buf, sw, sh, px as usize, py as usize, 6, c);
            }

            // Click on validation target → capture sample
            let val_mouse_down = window.get_mouse_down(MouseButton::Left);
            let val_mouse_edge = val_mouse_down && !prev_val_mouse_down;
            prev_val_mouse_down = val_mouse_down;

            let mp = window.get_mouse_pos(MouseMode::Discard).unwrap_or((0.0, 0.0));
            let click_dist = ((mp.0 as f64 - vtx).powi(2) + (mp.1 as f64 - vty).powi(2)).sqrt();
            if val_mouse_edge && click_dist < 80.0 {
                if let Some(feats) = &current_features {
                    if let Some((raw_x, raw_y)) = ridge_reg.predict(feats) {
                        let px = (raw_x as f64).clamp(0.0, sw as f64 - 1.0);
                        let py = (raw_y as f64).clamp(0.0, sh as f64 - 1.0);
                        validation_results.push((px, py, vtx, vty));
                        session.validation.push(ValidFrame {
                            features: feats.clone(),
                            target_x: vtx as f32,
                            target_y: vty as f32,
                        });
                        let err = ((px - vtx).powi(2) + (py - vty).powi(2)).sqrt();
                        println!("  Validation point {}/{}: predicted ({px:.0}, {py:.0}), target ({vtx:.0}, {vty:.0}), error {err:.0} px",
                            validation_idx + 1, validation_targets.len(), );
                    }
                    validation_idx += 1;
                }
            }

            // After all validation points clicked, compute aggregate
            if validation_idx >= validation_targets.len() {
                let errors: Vec<f64> = validation_results.iter()
                    .map(|(px, py, tx, ty)| ((px-tx).powi(2)+(py-ty).powi(2)).sqrt())
                    .collect();
                if !errors.is_empty() {
                    let n = errors.len() as f64;
                    let mean_err = errors.iter().sum::<f64>() / n;
                    let max_err = errors.iter().cloned().fold(0.0f64, f64::max);
                    let mut sorted = errors.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = sorted[sorted.len() / 2];

                    println!("\n=== Multi-point validation ===");
                    println!("  Points: {}", errors.len());
                    println!("  Mean error:   {mean_err:.0} px");
                    println!("  Median error: {median:.0} px");
                    println!("  Max error:    {max_err:.0} px");

                    // Compute coverage: range of predictions vs range of targets
                    let p_x: Vec<f64> = validation_results.iter().map(|r| r.0).collect();
                    let p_y: Vec<f64> = validation_results.iter().map(|r| r.1).collect();
                    let t_x: Vec<f64> = validation_results.iter().map(|r| r.2).collect();
                    let t_y: Vec<f64> = validation_results.iter().map(|r| r.3).collect();
                    let p_xr = p_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - p_x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let p_yr = p_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - p_y.iter().cloned().fold(f64::INFINITY, f64::min);
                    let t_xr = t_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t_x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let t_yr = t_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t_y.iter().cloned().fold(f64::INFINITY, f64::min);
                    println!("  Coverage X: {:.0}/{:.0} px ({:.0}%)", p_xr, t_xr, 100.0 * p_xr / t_xr.max(1.0));
                    println!("  Coverage Y: {:.0}/{:.0} px ({:.0}%)", p_yr, t_yr, 100.0 * p_yr / t_yr.max(1.0));
                    println!("  Press C to recalibrate, ESC to quit.");
                }

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
            // Continuous learning: every click adds a training sample at click coords
            // (assumes user looks where they click — WebGazer's key insight)
            if mouse_edge {
                if let Some(feats) = &current_features {
                    ridge_reg.add_sample(feats.clone(), mouse_pos.0 as f32, mouse_pos.1 as f32);
                    println!("  +1 click sample at ({:.0}, {:.0}), total: {}",
                        mouse_pos.0, mouse_pos.1, ridge_reg.sample_count());
                }
            }

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

/// Eye Aspect Ratio for blink detection (Soukupová & Čech 2016).
/// EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
/// Open eye: ~0.25-0.30. Closed: <0.15.
fn eye_aspect_ratio(lm: &[(f32, f32)], start: usize) -> f32 {
    let d = |a: (f32,f32), b: (f32,f32)| ((a.0-b.0).powi(2)+(a.1-b.1).powi(2)).sqrt();
    let v1 = d(lm[start+1], lm[start+5]);
    let v2 = d(lm[start+2], lm[start+4]);
    let h  = d(lm[start],   lm[start+3]);
    (v1 + v2) / (2.0 * h.max(1.0))
}
