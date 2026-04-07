//! CNN-based webcam eye tracker using MobileGaze (pretrained gaze direction CNN).
//!
//! Pipeline:
//!   camera → rustface (face bbox) → MobileGaze CNN → (yaw, pitch) angles
//!   → calibration: linear (yaw,pitch) → (screen_x, screen_y) mapping (6 params)
//!   → smoothing → cursor
//!
//! Why CNN: MobileGaze is trained on thousands of faces with ground-truth gaze.
//! It understands gaze direction in absolute terms, not pixel positions.
//! Should be more robust to head movement than pixel-feature ridge regression.
//!
//! Calibration is much simpler: just 6 parameters to fit, so 5-9 points is enough.

use display_info::DisplayInfo;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use nalgebra::{Matrix3, Vector3};
use saccade::calib_state::{CalibrationState, EventResult, Phase};
use saccade::one_euro::OneEuroFilter2D;
use saccade::sugano;
use std::time::Instant;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const FACE_M: &str = "seeta_fd_frontal_v1.0.bin";
const FACE_U: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";
const PFLD_M: &str = "pfld.onnx";
const PFLD_U: &str = "https://github.com/cunjian/pytorch_face_landmark/raw/refs/heads/master/onnx/pfld.onnx";
const GAZE_M: &str = "resnet18_gaze.onnx";
const GAZE_U: &str = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18_gaze.onnx";

fn dl(p: &str, u: &str) {
    if !std::path::Path::new(p).exists() {
        println!("Downloading {p}...");
        let _ = std::process::Command::new("curl").args(["-L", "-o", p, u]).status();
    }
}
fn load(p: &str) -> Model {
    tract_onnx::onnx().model_for_path(p).unwrap().into_optimized().unwrap().into_runnable().unwrap()
}

fn calib_points(w: usize, h: usize) -> Vec<(f64, f64)> {
    let mx = w as f64 * 0.1;
    let my = h as f64 * 0.1;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    vec![
        (mx, my), (cx, my), (w as f64 - mx, my),
        (mx, cy), (w as f64 - mx, cy),
        (mx, h as f64 - my), (cx, h as f64 - my), (w as f64 - mx, h as f64 - my),
        (cx, cy),
    ]
}

/// 2nd-degree polynomial on (yaw, pitch) → (screen_x, screen_y).
/// 6 features: [1, yaw, pitch, yaw², pitch², yaw·pitch]
/// Standard recipe from gaze estimation literature.
const POLY_FEAT_LEN: usize = 6;

#[derive(Clone, Copy)]
struct CnnSample {
    yaw: f32,
    pitch: f32,
    head_x: f32,
    head_y: f32,
    head_w: f32,
    head_roll: f32,
    screen_x: f32,
    screen_y: f32,
}

struct LinearMapper {
    samples: Vec<CnnSample>,
    coeffs_x: [f64; POLY_FEAT_LEN],
    coeffs_y: [f64; POLY_FEAT_LEN],
    fitted: bool,
    lambda: f64,
}

fn poly_features(s: &CnnSample) -> [f64; POLY_FEAT_LEN] {
    let yaw = s.yaw as f64;
    let pitch = s.pitch as f64;
    [
        1.0,
        yaw,
        pitch,
        yaw * yaw,
        pitch * pitch,
        yaw * pitch,
    ]
}

impl LinearMapper {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            coeffs_x: [0.0; POLY_FEAT_LEN],
            coeffs_y: [0.0; POLY_FEAT_LEN],
            fitted: false,
            lambda: 1.0,
        }
    }

    fn add(&mut self, yaw: f32, pitch: f32, head_x: f32, head_y: f32, head_w: f32, head_roll: f32, sx: f32, sy: f32) {
        self.samples.push(CnnSample { yaw, pitch, head_x, head_y, head_w, head_roll, screen_x: sx, screen_y: sy });
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.fitted = false;
    }

    /// Fit ridge regression on polynomial features via nalgebra.
    fn fit(&mut self) -> bool {
        use nalgebra::{DMatrix, DVector};
        let n = self.samples.len();
        if n < 3 { return false; }
        let p = POLY_FEAT_LEN;

        let mut x_data = Vec::with_capacity(n * p);
        let mut y_x = Vec::with_capacity(n);
        let mut y_y = Vec::with_capacity(n);
        for s in &self.samples {
            let f = poly_features(s);
            x_data.extend_from_slice(&f);
            y_x.push(s.screen_x as f64);
            y_y.push(s.screen_y as f64);
        }

        let x_mat = DMatrix::from_row_slice(n, p, &x_data);
        let xt = x_mat.transpose();
        let mut xtx = &xt * &x_mat;
        for i in 0..p { xtx[(i, i)] += self.lambda; }

        let xty_x = &xt * DVector::from_vec(y_x);
        let xty_y = &xt * DVector::from_vec(y_y);
        let decomp = xtx.lu();
        if let (Some(beta_x), Some(beta_y)) = (decomp.solve(&xty_x), decomp.solve(&xty_y)) {
            for i in 0..p {
                self.coeffs_x[i] = beta_x[i];
                self.coeffs_y[i] = beta_y[i];
            }
            self.fitted = true;
            true
        } else {
            false
        }
    }

    /// Auto-tune lambda via leave-one-out CV.
    fn auto_tune(&mut self) {
        let candidates = [1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6];
        let mut best = self.lambda;
        let mut best_err = f64::INFINITY;
        for &lam in &candidates {
            self.lambda = lam;
            let err = self.loo_error();
            if err < best_err { best_err = err; best = lam; }
        }
        self.lambda = best;
        self.fit();
    }

    fn predict(&self, yaw: f32, pitch: f32, head_x: f32, head_y: f32, head_w: f32, head_roll: f32) -> Option<(f32, f32)> {
        if !self.fitted { return None; }
        let s = CnnSample { yaw, pitch, head_x, head_y, head_w, head_roll, screen_x: 0.0, screen_y: 0.0 };
        let f = poly_features(&s);
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        for i in 0..POLY_FEAT_LEN {
            sx += self.coeffs_x[i] * f[i];
            sy += self.coeffs_y[i] * f[i];
        }
        Some((sx as f32, sy as f32))
    }

    fn loo_error(&self) -> f64 {
        let n = self.samples.len();
        if n < 4 { return f64::INFINITY; }
        let mut errors = Vec::new();
        for i in 0..n {
            let mut tmp = LinearMapper::new();
            tmp.lambda = self.lambda;
            for (j, s) in self.samples.iter().enumerate() {
                if j != i {
                    tmp.add(s.yaw, s.pitch, s.head_x, s.head_y, s.head_w, s.head_roll, s.screen_x, s.screen_y);
                }
            }
            if !tmp.fit() { continue; }
            let held = &self.samples[i];
            if let Some((px, py)) = tmp.predict(held.yaw, held.pitch, held.head_x, held.head_y, held.head_w, held.head_roll) {
                let dx = px as f64 - held.screen_x as f64;
                let dy = py as f64 - held.screen_y as f64;
                errors.push((dx*dx + dy*dy).sqrt());
            }
        }
        if errors.is_empty() { f64::INFINITY }
        else { errors.iter().sum::<f64>() / errors.len() as f64 }
    }
}

fn main() {
    dl(FACE_M, FACE_U);
    dl(PFLD_M, PFLD_U);
    dl(GAZE_M, GAZE_U);

    let mut face_det = rustface::create_detector(FACE_M).expect("face model");
    face_det.set_min_face_size(80);
    face_det.set_score_thresh(2.0);
    let pfld = load(PFLD_M);
    let gaze_net = load(GAZE_M);

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = loop {
        match Camera::new(CameraIndex::Index(0), format.clone()) {
            Ok(c) => break c,
            Err(e) => { eprintln!("Camera: {e}. Retry..."); std::thread::sleep(std::time::Duration::from_secs(2)); }
        }
    };
    camera.open_stream().expect("stream");
    let cam_res = camera.resolution();
    let sd = (cam_res.width() / 720).max(1) as usize;
    let cw = cam_res.width() as usize / sd;
    let ch = cam_res.height() as usize / sd;
    println!("Camera: {}x{} -> {}x{}", cam_res.width(), cam_res.height(), cw, ch);

    let displays = DisplayInfo::all().expect("displays");
    let primary = displays.iter().find(|d| d.is_primary).unwrap_or(&displays[0]);
    let sw = primary.width as usize;
    let sh = primary.height as usize;
    println!("Screen: {sw}x{sh}");

    let mut window = Window::new(
        "Saccade CNN [click dots, ESC quit]",
        sw, sh,
        WindowOptions { ..WindowOptions::default() },
    ).unwrap();
    window.set_position(0, 0);
    window.set_target_fps(60);

    let mut mapper = LinearMapper::new();
    let mut one_euro = OneEuroFilter2D::new(1.0, 0.05, 1.0);

    let targets = calib_points(sw, sh);
    // 5 clicks per point — polynomial fit needs more samples to be stable
    const SAMPLES_PER_POINT: u32 = 5;
    let mut calib = CalibrationState::new(targets.len(), SAMPLES_PER_POINT);

    let validation_targets: Vec<(f64, f64)> = vec![
        (sw as f64 * 0.2, sh as f64 * 0.2),
        (sw as f64 * 0.8, sh as f64 * 0.2),
        (sw as f64 / 2.0, sh as f64 / 2.0),
        (sw as f64 * 0.2, sh as f64 * 0.8),
        (sw as f64 * 0.8, sh as f64 * 0.8),
    ];
    let mut validation_idx = 0usize;
    let mut validation_results: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut prev_val_mouse_down = false;

    let mut buf = vec![0u32; sw * sh];
    let mut face_sm = SmRect::new(0.3);
    let mut frame_n = 0u64;
    let mut last_gaze: Option<(f32, f32)> = None; // cached CNN output

    // Debug: save first N normalized crops to disk for visual inspection
    let mut debug_saves_remaining = 10u32;
    let _ = std::fs::create_dir_all("debug_crops");

    let pip_w = 240usize;
    let pip_h = pip_w * ch / cw;
    let start = Instant::now();
    let mut fps_c = FpsC::new();
    let mut prev_mouse_down = false;
    let mut auto_started = false;

    println!("Click each red dot {SAMPLES_PER_POINT}× to calibrate (only {} clicks total).",
        targets.len() * SAMPLES_PER_POINT as usize);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_ms = start.elapsed().as_millis() as u64;
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        let mouse_edge = mouse_down && !prev_mouse_down;
        prev_mouse_down = mouse_down;
        let mouse_pos = window.get_mouse_pos(MouseMode::Discard).unwrap_or((0.0, 0.0));

        if window.is_key_down(Key::C) {
            calib.restart();
            mapper.clear();
            one_euro.reset();
            println!("\nRecalibrating.");
        }

        if !auto_started && calib.phase() == Phase::Idle {
            calib.start();
            mapper.clear();
            auto_started = true;
            println!("Calibration auto-started.");
        }

        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(i) => i, Err(_) => continue },
            Err(_) => continue,
        };
        let (fw, fh) = (decoded.width() as usize, decoded.height() as usize);
        let rf = decoded.as_raw();
        let mut rgb = vec![0u8; cw*ch*3];
        let mut gray = vec![0u8; cw*ch];
        for y in 0..ch { for x in 0..cw {
            let (sx, sy) = (x*sd, y*sd);
            if sx<fw && sy<fh {
                let si=(sy*fw+sx)*3; let di=(y*cw+x)*3;
                rgb[di]=rf[si]; rgb[di+1]=rf[si+1]; rgb[di+2]=rf[si+2];
                gray[y*cw+x]=(0.299*rf[si] as f32+0.587*rf[si+1] as f32+0.114*rf[si+2] as f32) as u8;
            }
        }}

        frame_n += 1;
        let do_face = frame_n % 3 == 1 || !face_sm.init;
        let faces = if do_face {
            face_det.detect(&ImageData::new(&gray, cw as u32, ch as u32))
        } else { Vec::new() };
        let detected = faces.iter().max_by_key(|f| { let b=f.bbox(); b.width()*b.height() });

        for px in buf.iter_mut() { *px = 0xFFFFFF; }

        // PIP preview
        let pip_x = sw - pip_w - 10;
        let pip_y = 10;
        for py in 0..pip_h { for px in 0..pip_w {
            let sx2 = px * cw / pip_w;
            let sy2 = py * ch / pip_h;
            if sx2 < cw && sy2 < ch {
                let si = (sy2 * cw + sx2) * 3;
                buf[(pip_y + py) * sw + pip_x + px] = ((rgb[si] as u32)<<16)|((rgb[si+1] as u32)<<8)|rgb[si+2] as u32;
            }
        }}
        draw_rect(&mut buf, sw, sh, pip_x.saturating_sub(1), pip_y.saturating_sub(1), pip_w+2, pip_h+2, 0xFFFFFF);

        // current sample: (yaw, pitch, head_x, head_y, head_w, head_roll)
        let mut current_sample: Option<(f32, f32, f32, f32, f32, f32)> = None;
        if let Some(face) = detected {
            let bb = face.bbox();
            face_sm.update(bb.x().max(0) as f64, bb.y().max(0) as f64, bb.width() as f64, bb.height() as f64);
        }

        if face_sm.init {
            let (fx, fy, fwf, fhf) = face_sm.get();

            // Run PFLD landmarks
            if let Some(lm) = run_pfld(&pfld, &gray, cw, ch, fx, fy, fwf, fhf) {
                // 6 anchor landmarks for PnP (matching face_model_3d order):
                // 36, 39, 42, 45 (eye corners), 48, 54 (mouth corners)
                let img_pts: [(f64, f64); 6] = [
                    (lm[36].0 as f64, lm[36].1 as f64),
                    (lm[39].0 as f64, lm[39].1 as f64),
                    (lm[42].0 as f64, lm[42].1 as f64),
                    (lm[45].0 as f64, lm[45].1 as f64),
                    (lm[48].0 as f64, lm[48].1 as f64),
                    (lm[54].0 as f64, lm[54].1 as f64),
                ];

                // Solve PnP — focal length ~ image width (approximation for unknown camera)
                let focal = cw as f64;
                if let Some((rotation, translation)) =
                    sugano::solve_pnp(&img_pts, focal, cw as f64, ch as f64)
                {
                    // Eye center 3D = midpoint of eye corners in camera frame
                    // Use actual camera-frame eye center (rotated face model + translation)
                    let model = sugano::face_model_3d();
                    let r_eye_3d = rotation * model[0] + translation; // outer right
                    let l_eye_3d = rotation * model[3] + translation; // outer left
                    let eye_center_3d = (r_eye_3d + l_eye_3d) * 0.5;

                    // Run gaze every 2nd frame on normalized crop
                    let do_gaze = frame_n % 2 == 0 || last_gaze.is_none();
                    if do_gaze {
                        let normalized = sugano::normalize_eye_crop(
                            &rgb, cw, ch,
                            eye_center_3d,
                            rotation,
                            focal,
                            448, 448,
                            600.0,   // virtual distance: 600 mm
                            1600.0,  // virtual focal length
                        );

                        // DEBUG: save first N normalized crops + raw face crop for comparison
                        if debug_saves_remaining > 0 {
                            let idx = 10 - debug_saves_remaining;
                            // Save normalized crop
                            if let Some(img) = image::RgbImage::from_raw(448, 448, normalized.clone()) {
                                let _ = img.save(format!("debug_crops/normalized_{idx:02}.png"));
                            }
                            // Save raw face crop for comparison
                            let crop_w = fwf;
                            let crop_h = fhf;
                            let mut raw_crop = vec![0u8; crop_w * crop_h * 3];
                            for y in 0..crop_h {
                                for x in 0..crop_w {
                                    if fx + x < cw && fy + y < ch {
                                        let si = ((fy + y) * cw + (fx + x)) * 3;
                                        let di = (y * crop_w + x) * 3;
                                        raw_crop[di] = rgb[si];
                                        raw_crop[di+1] = rgb[si+1];
                                        raw_crop[di+2] = rgb[si+2];
                                    }
                                }
                            }
                            if let Some(img) = image::RgbImage::from_raw(crop_w as u32, crop_h as u32, raw_crop) {
                                let _ = img.save(format!("debug_crops/raw_face_{idx:02}.png"));
                            }
                            // Also dump diagnostic info
                            println!("[DEBUG] Save {idx:02}: face=({fx},{fy},{fwf},{fhf}) eye_3d=({:.1},{:.1},{:.1}) tz={:.1}",
                                eye_center_3d.x, eye_center_3d.y, eye_center_3d.z, translation.z);
                            debug_saves_remaining -= 1;
                            if debug_saves_remaining == 0 {
                                println!("[DEBUG] Finished saving 10 crops to debug_crops/");
                            }
                        }

                        if let Some((yaw_n, pitch_n)) = run_gaze_from_buffer(&gaze_net, &normalized, 448, 448) {
                            // Denormalize back to real camera frame
                            let r_norm = build_normalization_rotation(&rotation, &eye_center_3d);
                            let (yaw, pitch) = sugano::denormalize_gaze(yaw_n, pitch_n, &r_norm);
                            last_gaze = Some((yaw, pitch));
                        }
                    }
                }
            }

            // Head pose features (still useful for polynomial mapper)
            let head_x = (fx as f32 + fwf as f32 / 2.0) / cw as f32 * 100.0;
            let head_y = (fy as f32 + fhf as f32 / 2.0) / ch as f32 * 100.0;
            let head_w = fwf as f32 / cw as f32 * 100.0;
            let head_roll = 0.0f32;

            if let Some((yaw, pitch)) = last_gaze {
                current_sample = Some((yaw, pitch, head_x, head_y, head_w, head_roll));
            }
        }

        // Calibration phase
        if calib.phase() == Phase::Calibrating {
            let calib_idx = calib.current_point();
            let samples_at = calib.samples_at_current();
            let (tx, ty) = targets[calib_idx];

            // White background — dark color saturates with clicks
            let progress = samples_at as f32 / SAMPLES_PER_POINT as f32;
            let red = (255.0 - 100.0 * progress) as u32;
            let green = (150.0 * (1.0 - progress)) as u32;
            let blue = (150.0 * (1.0 - progress)) as u32;
            let color = (red << 16) | (green << 8) | blue;
            let r = 22 + (progress * 8.0) as usize;
            draw_filled_circle(&mut buf, sw, sh, tx as usize, ty as usize, r, color);
            draw_ring(&mut buf, sw, sh, tx as usize, ty as usize, r + 3, 0x000000);

            let click_dist = ((mouse_pos.0 as f64 - tx).powi(2) + (mouse_pos.1 as f64 - ty).powi(2)).sqrt();
            if mouse_edge && click_dist < 80.0 {
                let has = current_sample.is_some();
                let result = calib.handle_capture(has);
                match result {
                    EventResult::SampleCaptured { point, sample } => {
                        if let Some((yaw, pitch, hx, hy, hw, hr)) = current_sample {
                            mapper.add(yaw, pitch, hx, hy, hw, hr, tx as f32, ty as f32);
                        }
                        println!("  Point {}/{} sample {}/{}", point + 1, targets.len(), sample, SAMPLES_PER_POINT);
                    }
                    EventResult::NextPoint { point } => {
                        if let Some((yaw, pitch, hx, hy, hw, hr)) = current_sample {
                            mapper.add(yaw, pitch, hx, hy, hw, hr, tx as f32, ty as f32);
                        }
                        println!("  Point {} done, next: {}/{}", calib_idx + 1, point + 1, targets.len());
                    }
                    EventResult::CalibrationComplete => {
                        if let Some((yaw, pitch, hx, hy, hw, hr)) = current_sample {
                            mapper.add(yaw, pitch, hx, hy, hw, hr, tx as f32, ty as f32);
                        }
                        // Auto-tune lambda via LOO CV
                        mapper.auto_tune();
                        let loo = mapper.loo_error();
                        println!("\nCalibration done: {} samples. λ={:.0e} LOO err: {loo:.0} px",
                            mapper.samples.len(), mapper.lambda);
                        validation_idx = 0;
                        validation_results.clear();
                        one_euro.reset();
                    }
                    _ => {}
                }
            }
        }

        // Validation phase
        if calib.phase() == Phase::Validating && validation_idx < validation_targets.len() {
            let (vtx, vty) = validation_targets[validation_idx];
            draw_filled_circle(&mut buf, sw, sh, vtx as usize, vty as usize, 24, 0x0066CC);
            draw_ring(&mut buf, sw, sh, vtx as usize, vty as usize, 30, 0x000000);
            for (i, &(px, py)) in validation_targets.iter().enumerate() {
                let c = if i < validation_idx { 0x009900 }
                        else if i == validation_idx { 0x0066CC }
                        else { 0xAAAAAA };
                draw_filled_circle(&mut buf, sw, sh, px as usize, py as usize, 6, c);
            }

            let val_md = window.get_mouse_down(MouseButton::Left);
            let val_edge = val_md && !prev_val_mouse_down;
            prev_val_mouse_down = val_md;

            let mp = window.get_mouse_pos(MouseMode::Discard).unwrap_or((0.0, 0.0));
            let click_dist = ((mp.0 as f64 - vtx).powi(2) + (mp.1 as f64 - vty).powi(2)).sqrt();
            if val_edge && click_dist < 80.0 {
                if let Some((yaw, pitch, hx, hy, hw, hr)) = current_sample {
                    if let Some((px, py)) = mapper.predict(yaw, pitch, hx, hy, hw, hr) {
                        let px = (px as f64).clamp(0.0, sw as f64 - 1.0);
                        let py = (py as f64).clamp(0.0, sh as f64 - 1.0);
                        validation_results.push((px, py, vtx, vty));
                        let err = ((px - vtx).powi(2) + (py - vty).powi(2)).sqrt();
                        println!("  Val {}/{}: pred ({px:.0}, {py:.0}), tgt ({vtx:.0}, {vty:.0}), err {err:.0}",
                            validation_idx + 1, validation_targets.len());
                    }
                }
                validation_idx += 1;
            }

            if validation_idx >= validation_targets.len() {
                let errors: Vec<f64> = validation_results.iter()
                    .map(|(px,py,tx,ty)| ((px-tx).powi(2)+(py-ty).powi(2)).sqrt()).collect();
                if !errors.is_empty() {
                    let n = errors.len() as f64;
                    let mean = errors.iter().sum::<f64>() / n;
                    let max = errors.iter().cloned().fold(0.0f64, f64::max);
                    let mut sorted = errors.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = sorted[sorted.len()/2];

                    let p_x: Vec<f64> = validation_results.iter().map(|r| r.0).collect();
                    let p_y: Vec<f64> = validation_results.iter().map(|r| r.1).collect();
                    let t_x: Vec<f64> = validation_results.iter().map(|r| r.2).collect();
                    let t_y: Vec<f64> = validation_results.iter().map(|r| r.3).collect();
                    let p_xr = p_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - p_x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let p_yr = p_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - p_y.iter().cloned().fold(f64::INFINITY, f64::min);
                    let t_xr = t_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t_x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let t_yr = t_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t_y.iter().cloned().fold(f64::INFINITY, f64::min);

                    println!("\n=== CNN Multi-point validation ===");
                    println!("  Points: {}", errors.len());
                    println!("  Mean:   {mean:.0} px");
                    println!("  Median: {median:.0} px");
                    println!("  Max:    {max:.0} px");
                    println!("  Coverage X: {:.0}/{:.0} ({:.0}%)", p_xr, t_xr, 100.0 * p_xr / t_xr.max(1.0));
                    println!("  Coverage Y: {:.0}/{:.0} ({:.0}%)", p_yr, t_yr, 100.0 * p_yr / t_yr.max(1.0));
                }
                calib.finish_validation();
            }
        }

        // Running phase
        if calib.phase() == Phase::Running {
            if mouse_edge {
                if let Some((yaw, pitch, hx, hy, hw, hr)) = current_sample {
                    mapper.add(yaw, pitch, hx, hy, hw, hr, mouse_pos.0 as f32, mouse_pos.1 as f32);
                    mapper.fit();
                    println!("  +1 sample, total {}", mapper.samples.len());
                }
            }
            if let Some((yaw, pitch, hx, hy, hw, hr)) = current_sample {
                if let Some((px, py)) = mapper.predict(yaw, pitch, hx, hy, hw, hr) {
                    let cx = (px as f64).clamp(0.0, sw as f64 - 1.0);
                    let cy = (py as f64).clamp(0.0, sh as f64 - 1.0);
                    let t_sec = now_ms as f64 / 1000.0;
                    let (gx_f, gy_f) = one_euro.filter((cx, cy), t_sec);
                    let gx = gx_f as usize;
                    let gy = gy_f as usize;
                    draw_filled_circle(&mut buf, sw, sh, gx, gy, 10, 0x00FF00);
                    draw_ring(&mut buf, sw, sh, gx, gy, 20, 0xFFFFFF);
                    draw_ring(&mut buf, sw, sh, gx, gy, 30, 0x00FF00);
                }
            }
        }

        fps_c.tick();
        let f = fps_c.fps();
        let bar = (f as usize * 5).min(sw);
        let bc = if f > 15.0 { 0x00FF00 } else if f > 8.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar { for y in 0..4 { buf[y*sw+x] = bc; } }
        window.update_with_buffer(&buf, sw, sh).unwrap();

        if fps_c.count % 30 == 0 {
            let mode = match calib.phase() {
                Phase::Idle => "IDLE", Phase::Calibrating => "CALIB",
                Phase::Validating => "VALID", Phase::Running => "RUN",
            };
            print!("\r[{mode}] FPS:{f:.0} | samples:{} | gaze:{}    ",
                mapper.samples.len(),
                if current_sample.is_some() { "ok" } else { "?" });
        }
    }
    println!("\nDone.");
}

/// Build the normalization rotation R_n that aligns the eye center to camera z-axis.
/// (Same as inside sugano::normalize_eye_crop, but exposed for denormalization.)
fn build_normalization_rotation(rotation: &Matrix3<f64>, eye_center_3d: &Vector3<f64>) -> Matrix3<f64> {
    let z_axis = eye_center_3d / eye_center_3d.norm();
    let head_x = rotation.column(0).into_owned();
    let mut x_axis = head_x - z_axis * head_x.dot(&z_axis);
    let xn = x_axis.norm();
    if xn < 1e-6 {
        x_axis = Vector3::new(1.0, 0.0, 0.0);
    } else {
        x_axis /= xn;
    }
    let y_axis = z_axis.cross(&x_axis);
    Matrix3::from_columns(&[x_axis, y_axis, z_axis]).transpose()
}

/// Run gaze CNN on a pre-normalized RGB buffer (already 448×448).
fn run_gaze_from_buffer(model: &Model, rgb: &[u8], w: usize, h: usize) -> Option<(f32, f32)> {
    if w != 448 || h != 448 || rgb.len() != w * h * 3 { return None; }
    let sz = 448;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut data = vec![0.0f32; 3*sz*sz];
    for y in 0..sz { for x in 0..sz {
        let si = (y*sz+x)*3;
        for c in 0..3 { data[c*sz*sz+y*sz+x] = (rgb[si+c] as f32 / 255.0 - mean[c]) / std[c]; }
    }}
    let t = Tensor::from(tract_ndarray::Array4::from_shape_vec((1,3,sz,sz), data).ok()?).into();
    let r = model.run(tvec![t]).ok()?;
    let yaw_view = r[0].to_array_view::<f32>().ok()?;
    let pitch_view = r[1].to_array_view::<f32>().ok()?;
    Some((bins_to_angle(yaw_view.as_slice()?), bins_to_angle(pitch_view.as_slice()?)))
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

fn run_gaze(model: &Model, rgb: &[u8], w: usize, h: usize, fx: usize, fy: usize, fw: usize, fh: usize) -> Option<(f32, f32)> {
    if fx+fw > w || fy+fh > h || fw < 20 || fh < 20 { return None; }
    let sz = 448;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut data = vec![0.0f32; 3*sz*sz];
    for y in 0..sz { for x in 0..sz {
        let sx = fx + x*fw/sz;
        let sy = fy + y*fh/sz;
        if sx < w && sy < h {
            let si = (sy*w+sx)*3;
            for c in 0..3 { data[c*sz*sz+y*sz+x] = (rgb[si+c] as f32 / 255.0 - mean[c]) / std[c]; }
        }
    }}
    let t = Tensor::from(tract_ndarray::Array4::from_shape_vec((1,3,sz,sz), data).ok()?).into();
    let r = model.run(tvec![t]).ok()?;
    let yaw_view = r[0].to_array_view::<f32>().ok()?;
    let pitch_view = r[1].to_array_view::<f32>().ok()?;
    let yaw_bins = yaw_view.as_slice()?;
    let pitch_bins = pitch_view.as_slice()?;
    Some((bins_to_angle(yaw_bins), bins_to_angle(pitch_bins)))
}

fn bins_to_angle(bins: &[f32]) -> f32 {
    let max_val = bins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = bins.iter().map(|&b| (b - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().enumerate().map(|(i, &e)| (e / sum) * (i as f32 * 4.0 - 180.0)).sum()
}

// --- UI helpers ---
struct SmRect{x:f64,y:f64,w:f64,h:f64,a:f64,init:bool}
impl SmRect{fn new(a:f64)->Self{Self{x:0.0,y:0.0,w:0.0,h:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64,w:f64,h:f64){if!self.init{self.x=x;self.y=y;self.w=w;self.h=h;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;self.w=a*w+(1.0-a)*self.w;self.h=a*h+(1.0-a)*self.h;}}fn get(&self)->(usize,usize,usize,usize){(self.x.round()as usize,self.y.round()as usize,self.w.round().max(1.0)as usize,self.h.round().max(1.0)as usize)}}
fn draw_filled_circle(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for dy in 0..=r{for dx in 0..=r{if dx*dx+dy*dy<=r*r{for&(sx,sy)in&[(cx+dx,cy+dy),(cx.wrapping_sub(dx),cy+dy),(cx+dx,cy.wrapping_sub(dy)),(cx.wrapping_sub(dx),cy.wrapping_sub(dy))]{if sx<w&&sy<h{b[sy*w+sx]=c;}}}}}}
fn draw_ring(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for i in 0..64{let t=2.0*std::f64::consts::PI*i as f64/64.0;let x=(cx as f64+r as f64*t.cos()).round()as i32;let y=(cy as f64+r as f64*t.sin()).round()as i32;if x>=0&&(x as usize)<w&&y>=0&&(y as usize)<h{b[y as usize*w+x as usize]=c;}}}
fn draw_rect(b:&mut[u32],w:usize,h:usize,x:usize,y:usize,rw:usize,rh:usize,c:u32){for dx in 0..rw{let px=x+dx;if px<w{if y<h{b[y*w+px]=c;}let by=y+rh.saturating_sub(1);if by<h{b[by*w+px]=c;}}}for dy in 0..rh{let py=y+dy;if py<h{if x<w{b[py*w+x]=c;}let bx=x+rw.saturating_sub(1);if bx<w{b[py*w+bx]=c;}}}}
struct FpsC{t:Instant,count:u64,fps:f64}
impl FpsC{fn new()->Self{Self{t:Instant::now(),count:0,fps:0.0}}fn tick(&mut self){self.count+=1;let e=self.t.elapsed().as_secs_f64();if e>=1.0{self.fps=self.count as f64/e;self.count=0;self.t=Instant::now();}}fn fps(&self)->f64{self.fps}}
