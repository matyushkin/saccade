//! Real-time eye tracking from webcam with EAR-based blink detection.
//!
//! Pipeline: nokhwa (camera) → rustface (face bbox) → PFLD ONNX (68 landmarks)
//!          → Timm & Barth (pupil center) → EAR (blink detection) → minifb (display)
//!
//! Usage:
//!   cargo run --release --features demo --example webcam
//!
//! Downloads face detection model and landmark model on first run.

use minifb::{Key, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use ort::session::Session;
use rustface::ImageData;
use saccade::blink::{BlinkDetector, EyeState};
use saccade::classify::{EyeEvent, IVTClassifier};
use saccade::ear;
use saccade::frame::GrayFrame;
use saccade::timm::{self, TimmConfig};
use std::time::Instant;

const FACE_MODEL_PATH: &str = "seeta_fd_frontal_v1.0.bin";
const FACE_MODEL_URL: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";
const PFLD_MODEL_PATH: &str = "pfld.onnx";
const PFLD_MODEL_URL: &str = "https://github.com/cunjian/pytorch_face_landmark/raw/refs/heads/master/onnx/pfld.onnx";

fn download_if_missing(path: &str, url: &str) {
    if !std::path::Path::new(path).exists() {
        println!("Downloading {path}...");
        let status = std::process::Command::new("curl")
            .args(["-L", "-o", path, url])
            .status()
            .expect("Failed to run curl");
        if !status.success() {
            eprintln!("Failed to download. Please run: curl -L -o {path} {url}");
            std::process::exit(1);
        }
    }
}

fn main() {
    download_if_missing(FACE_MODEL_PATH, FACE_MODEL_URL);
    download_if_missing(PFLD_MODEL_PATH, PFLD_MODEL_URL);

    // Face detector
    let mut detector = rustface::create_detector(FACE_MODEL_PATH)
        .expect("Failed to load face model");
    detector.set_min_face_size(80);
    detector.set_score_thresh(2.0);

    // PFLD landmark model (68 points, input 112×112)
    let mut landmark_session = Session::builder()
        .expect("Failed to create ONNX session builder")
        .commit_from_file(PFLD_MODEL_PATH)
        .expect("Failed to load PFLD model");

    // Camera
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(0), format).expect("Failed to open camera");
    camera.open_stream().expect("Failed to open stream");

    let cam_res = camera.resolution();
    let scale_down = (cam_res.width() / 640).max(1) as usize;
    let cam_w = cam_res.width() as usize / scale_down;
    let cam_h = cam_res.height() as usize / scale_down;
    println!("Camera: {}×{} → {}×{}", cam_res.width(), cam_res.height(), cam_w, cam_h);

    // Window
    let mut window = Window::new("Saccade — Eye Tracker (ESC to quit)", cam_w, cam_h, WindowOptions::default())
        .expect("Failed to create window");
    window.set_target_fps(60);

    // Timm config
    let timm_config = TimmConfig {
        gradient_threshold: 0.2,
        use_weight_map: true,
        weight_blur_sigma: 2.0,
    };

    // Smoothers
    let mut face_smooth = SmoothRect::new(0.25);
    let mut left_pupil_smooth = SmoothPoint::new(0.4);
    let mut right_pupil_smooth = SmoothPoint::new(0.4);
    let mut left_ear_smooth = 0.3f64;
    let mut right_ear_smooth = 0.3f64;

    // Blink detectors — hybrid openness threshold
    let mut left_blink = BlinkDetector::new();
    left_blink.confidence_threshold = 0.5;
    let mut right_blink = BlinkDetector::new();
    right_blink.confidence_threshold = 0.5;

    let mut classifier = IVTClassifier::default_params();
    let start_time = Instant::now();

    let mut frame_buf = vec![0u32; cam_w * cam_h];
    let mut fps_counter = FpsCounter::new();
    let mut no_face_count = 0u32;

    // Pre-allocate PFLD input buffer (1×3×112×112)
    let pfld_size = 112usize;

    println!("Running... Press ESC to quit.");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_ms = start_time.elapsed().as_millis() as u64;

        // Capture
        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(img) => img, Err(_) => continue },
            Err(_) => continue,
        };
        let full_w = decoded.width() as usize;
        let full_h = decoded.height() as usize;
        let rgb_full = decoded.as_raw();

        // Downscale
        let mut rgb_data = vec![0u8; cam_w * cam_h * 3];
        let mut gray = vec![0u8; cam_w * cam_h];
        for y in 0..cam_h {
            for x in 0..cam_w {
                let sx = x * scale_down;
                let sy = y * scale_down;
                if sx < full_w && sy < full_h {
                    let si = (sy * full_w + sx) * 3;
                    let di = (y * cam_w + x) * 3;
                    let (r, g, b) = (rgb_full[si], rgb_full[si + 1], rgb_full[si + 2]);
                    rgb_data[di] = r;
                    rgb_data[di + 1] = g;
                    rgb_data[di + 2] = b;
                    gray[y * cam_w + x] = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                }
            }
        }

        // Detect faces
        let image_data = ImageData::new(&gray, cam_w as u32, cam_h as u32);
        let faces = detector.detect(&image_data);

        // RGB → display buffer
        for i in 0..cam_w * cam_h {
            frame_buf[i] = ((rgb_data[i*3] as u32) << 16) | ((rgb_data[i*3+1] as u32) << 8) | rgb_data[i*3+2] as u32;
        }

        let best_face = faces.iter().max_by_key(|f| {
            let b = f.bbox();
            b.width() as i32 * b.height() as i32
        });

        if let Some(face) = best_face {
            no_face_count = 0;
            let bbox = face.bbox();
            face_smooth.update(
                bbox.x().max(0) as f64, bbox.y().max(0) as f64,
                bbox.width() as f64, bbox.height() as f64,
            );
            let (sfx, sfy, sfw, sfh) = face_smooth.get();
            draw_rect(&mut frame_buf, cam_w, cam_h, sfx, sfy, sfw, sfh, 0xFFFF00);

            // Run PFLD on the face crop
            if let Some(landmarks) = run_pfld(&mut landmark_session, &gray, cam_w, cam_h, sfx, sfy, sfw, sfh, pfld_size) {
                // Draw all 68 landmarks as tiny dots
                for &(lx, ly) in &landmarks {
                    let px = lx.round() as usize;
                    let py = ly.round() as usize;
                    if px < cam_w && py < cam_h {
                        frame_buf[py * cam_w + px] = 0x00FF00;
                    }
                }

                // Compute EAR + dark_ratio hybrid for each eye
                let ear_pair = ear::compute_ear_from_landmarks(&landmarks);

                // Right eye (landmarks 36-41)
                let (right_open, right_pupil) = process_eye(
                    &landmarks, 36, 41, ear_pair.map(|p| p.0),
                    &gray, cam_w, cam_h, &timm_config,
                );
                right_ear_smooth = 0.4 * right_open + 0.6 * right_ear_smooth;
                let right_state = right_blink.update(right_ear_smooth, now_ms);
                if let Some((px, py)) = right_pupil {
                    right_pupil_smooth.update(px, py);
                    let (spx, spy) = right_pupil_smooth.get();
                    draw_crosshair(&mut frame_buf, cam_w, cam_h, spx, spy, 0xFF0000);
                    classifier.update(px, py, now_ms);
                }

                // Left eye (landmarks 42-47)
                let (left_open, left_pupil) = process_eye(
                    &landmarks, 42, 47, ear_pair.map(|p| p.1),
                    &gray, cam_w, cam_h, &timm_config,
                );
                left_ear_smooth = 0.4 * left_open + 0.6 * left_ear_smooth;
                let left_state = left_blink.update(left_ear_smooth, now_ms);
                if let Some((px, py)) = left_pupil {
                    left_pupil_smooth.update(px, py);
                    let (spx, spy) = left_pupil_smooth.get();
                    draw_crosshair(&mut frame_buf, cam_w, cam_h, spx, spy, 0x00FFFF);
                }

                // Eye state indicators — experimental (PFLD doesn't track closed eyes well)
                // Show as small dots, dimmed to indicate unreliable
                draw_filled_circle(&mut frame_buf, cam_w, cam_h, cam_w - 40, 15, 5, eye_state_color(right_state));
                draw_filled_circle(&mut frame_buf, cam_w, cam_h, cam_w - 20, 15, 5, eye_state_color(left_state));
            }
        } else {
            no_face_count += 1;
            if no_face_count > 30 {
                left_blink.reset();
                right_blink.reset();
                left_pupil_smooth = SmoothPoint::new(0.4);
                right_pupil_smooth = SmoothPoint::new(0.4);
                no_face_count = 0;
            } else if face_smooth.initialized {
                let (sfx, sfy, sfw, sfh) = face_smooth.get();
                draw_rect(&mut frame_buf, cam_w, cam_h, sfx, sfy, sfw, sfh, 0x666600);
            }
        }

        // FPS bar
        fps_counter.tick();
        let fps = fps_counter.fps();
        let bar_len = (fps as usize * 3).min(cam_w);
        let bar_color = if fps > 20.0 { 0x00FF00 } else if fps > 10.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar_len {
            for y in 0..4 { frame_buf[y * cam_w + x] = bar_color; }
        }

        window.update_with_buffer(&frame_buf, cam_w, cam_h).unwrap();

        if fps_counter.frame_count % 30 == 0 {
            let l_state = format!("{:?}", left_blink.state());
            let r_state = format!("{:?}", right_blink.state());
            let blinks = left_blink.blink_count() + right_blink.blink_count();
            let bpm = (left_blink.blinks_per_minute(now_ms, 60_000)
                + right_blink.blinks_per_minute(now_ms, 60_000)) / 2.0;
            let fixations = classifier.events().iter().filter(|e| matches!(e, EyeEvent::Fixation(_))).count();
            let saccades = classifier.events().iter().filter(|e| matches!(e, EyeEvent::Saccade(_))).count();
            print!("\rFPS:{fps:.0} | EAR L:{left_ear_smooth:.2} R:{right_ear_smooth:.2} | {l_state}/{r_state} | Blinks:{blinks} ({bpm:.0}/min) | Fix:{fixations} Sac:{saccades}    ");
        }
    }
    println!("\nDone.");
}

/// Process one eye: compute hybrid openness (EAR + dark_ratio) and detect pupil.
/// Returns (openness, Option<(pupil_x, pupil_y)>).
fn process_eye(
    landmarks: &[(f32, f32)],
    start_idx: usize,
    end_idx: usize,
    ear: Option<f32>,
    gray: &[u8],
    img_w: usize,
    img_h: usize,
    config: &TimmConfig,
) -> (f64, Option<(f64, f64)>) {
    // Get eye ROI from landmarks
    let eye_points = &landmarks[start_idx..=end_idx];
    let min_x = eye_points.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let max_x = eye_points.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let min_y = eye_points.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let max_y = eye_points.iter().map(|p| p.1).fold(f32::MIN, f32::max);

    let margin_x = (max_x - min_x) * 0.3;
    let margin_y = (max_y - min_y) * 0.5;
    let roi_x = (min_x - margin_x).max(0.0) as usize;
    let roi_y = (min_y - margin_y).max(0.0) as usize;
    let roi_w = ((max_x - min_x) + 2.0 * margin_x) as usize;
    let roi_h = ((max_y - min_y) + 2.0 * margin_y) as usize;

    if roi_x + roi_w > img_w || roi_y + roi_h > img_h || roi_w < 8 || roi_h < 6 {
        return (0.0, None);
    }

    // Compute dark_ratio in eye ROI
    let mut eye_data = vec![0u8; roi_w * roi_h];
    for y in 0..roi_h {
        let src = (roi_y + y) * img_w + roi_x;
        eye_data[y * roi_w..(y + 1) * roi_w].copy_from_slice(&gray[src..src + roi_w]);
    }

    let n = eye_data.len() as f64;
    let mean = eye_data.iter().map(|&p| p as f64).sum::<f64>() / n;
    let dark_thresh = (mean * 0.6) as u8;
    let dark_count = eye_data.iter().filter(|&&p| p < dark_thresh).count();
    let dark_ratio = dark_count as f64 / n;

    // Hybrid openness: combine EAR (if available) with dark_ratio
    // EAR: ~0.30 open, ~0.15 blink — but PFLD may not track closed eyes well
    // dark_ratio: ~0.08-0.12 open, ~0.02-0.04 closed
    // Use dark_ratio as the primary signal, EAR as bonus
    let ear_signal = ear.map(|e| e as f64).unwrap_or(0.3);
    let openness = 0.4 * (dark_ratio / 0.10).clamp(0.0, 1.0) + 0.6 * (ear_signal / 0.30).clamp(0.0, 1.0);

    // Detect pupil if eye seems open
    let pupil = if openness > 0.4 {
        detect_pupil_in_roi(&mut eye_data, roi_x, roi_y, roi_w, roi_h, config)
    } else {
        None
    };

    (openness, pupil)
}

/// Detect pupil center using Timm & Barth in a pre-extracted eye ROI.
fn detect_pupil_in_roi(
    eye_data: &mut [u8],
    roi_x: usize, roi_y: usize,
    roi_w: usize, roi_h: usize,
    config: &TimmConfig,
) -> Option<(f64, f64)> {
    // Contrast stretch
    let min_val = *eye_data.iter().min().unwrap_or(&0);
    let max_val = *eye_data.iter().max().unwrap_or(&255);
    if max_val > min_val + 10 {
        let range = (max_val - min_val) as f32;
        for p in eye_data.iter_mut() {
            *p = ((*p as f32 - min_val as f32) / range * 255.0) as u8;
        }
    }

    let frame = GrayFrame::new(roi_w as u32, roi_h as u32, eye_data);
    let result = timm::detect_center(&frame, config);
    Some((roi_x as f64 + result.x, roi_y as f64 + result.y))
}

/// Run PFLD landmark model on a face crop. Returns 68 landmarks in original image coordinates.
fn run_pfld(
    session: &mut Session,
    gray: &[u8], img_w: usize, img_h: usize,
    fx: usize, fy: usize, fw: usize, fh: usize,
    pfld_size: usize,
) -> Option<Vec<(f32, f32)>> {
    if fx + fw > img_w || fy + fh > img_h || fw < 20 || fh < 20 {
        return None;
    }

    // Prepare 112×112 RGB input from face crop (grayscale → 3-channel)
    let mut input = vec![0.0f32; 3 * pfld_size * pfld_size];
    for y in 0..pfld_size {
        for x in 0..pfld_size {
            let src_x = fx + x * fw / pfld_size;
            let src_y = fy + y * fh / pfld_size;
            let val = if src_x < img_w && src_y < img_h {
                gray[src_y * img_w + src_x] as f32 / 255.0
            } else {
                0.0
            };
            // NCHW format: [batch, channel, height, width]
            input[0 * pfld_size * pfld_size + y * pfld_size + x] = val; // R
            input[1 * pfld_size * pfld_size + y * pfld_size + x] = val; // G
            input[2 * pfld_size * pfld_size + y * pfld_size + x] = val; // B
        }
    }

    let input_tensor = ort::value::Tensor::from_array(([1usize, 3, pfld_size, pfld_size], input.into_boxed_slice())).ok()?;
    let outputs = session.run(ort::inputs![input_tensor]).ok()?;

    // Output: (1, 136) — 68 landmarks × 2 (x, y) normalized to [0, 1]
    let output_val = outputs.iter().next()?.1;
    let (_, data) = output_val.try_extract_tensor::<f32>().ok()?;

    if data.len() < 136 {
        return None;
    }

    let landmarks: Vec<(f32, f32)> = (0..68)
        .map(|i| {
            let lx = data[i * 2] * fw as f32 + fx as f32;
            let ly = data[i * 2 + 1] * fh as f32 + fy as f32;
            (lx, ly)
        })
        .collect();

    Some(landmarks)
}


// --- UI helpers ---

struct SmoothRect { x: f64, y: f64, w: f64, h: f64, alpha: f64, initialized: bool }
impl SmoothRect {
    fn new(alpha: f64) -> Self { Self { x: 0.0, y: 0.0, w: 0.0, h: 0.0, alpha, initialized: false } }
    fn update(&mut self, x: f64, y: f64, w: f64, h: f64) {
        if !self.initialized { self.x = x; self.y = y; self.w = w; self.h = h; self.initialized = true; }
        else { let a = self.alpha; self.x = a*x+(1.0-a)*self.x; self.y = a*y+(1.0-a)*self.y; self.w = a*w+(1.0-a)*self.w; self.h = a*h+(1.0-a)*self.h; }
    }
    fn get(&self) -> (usize, usize, usize, usize) { (self.x.round() as usize, self.y.round() as usize, self.w.round().max(1.0) as usize, self.h.round().max(1.0) as usize) }
}

struct SmoothPoint { x: f64, y: f64, alpha: f64, initialized: bool }
impl SmoothPoint {
    fn new(alpha: f64) -> Self { Self { x: 0.0, y: 0.0, alpha, initialized: false } }
    fn update(&mut self, x: f64, y: f64) {
        if !self.initialized { self.x = x; self.y = y; self.initialized = true; }
        else { let a = self.alpha; self.x = a*x+(1.0-a)*self.x; self.y = a*y+(1.0-a)*self.y; }
    }
    fn get(&self) -> (usize, usize) { (self.x.round() as usize, self.y.round() as usize) }
}

fn eye_state_color(state: EyeState) -> u32 {
    match state { EyeState::Open => 0x00FF00, EyeState::Blinking => 0xFFFF00, EyeState::Closed => 0xFF0000 }
}

fn draw_filled_circle(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, r: usize, color: u32) {
    for dy in 0..=r { for dx in 0..=r { if dx*dx+dy*dy <= r*r {
        for &(sx,sy) in &[(cx+dx,cy+dy),(cx.wrapping_sub(dx),cy+dy),(cx+dx,cy.wrapping_sub(dy)),(cx.wrapping_sub(dx),cy.wrapping_sub(dy))] {
            if sx < w && sy < h { buf[sy*w+sx] = color; }
        }
    }}}
}

fn draw_rect(buf: &mut [u32], w: usize, h: usize, x: usize, y: usize, rw: usize, rh: usize, color: u32) {
    for dx in 0..rw { let px = x+dx; if px < w { if y < h { buf[y*w+px]=color; } let by=y+rh.saturating_sub(1); if by < h { buf[by*w+px]=color; } } }
    for dy in 0..rh { let py = y+dy; if py < h { if x < w { buf[py*w+x]=color; } let bx=x+rw.saturating_sub(1); if bx < w { buf[py*w+bx]=color; } } }
}

fn draw_crosshair(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, color: u32) {
    for d in -6i32..=6 {
        let x = cx as i32+d; if x >= 0 && (x as usize) < w && cy < h { buf[cy*w+x as usize]=color; }
        let y = cy as i32+d; if y >= 0 && (y as usize) < h && cx < w { buf[y as usize*w+cx]=color; }
    }
    for i in 0..48 { let t=2.0*std::f64::consts::PI*i as f64/48.0; let x=(cx as f64+8.0*t.cos()).round() as i32; let y=(cy as f64+8.0*t.sin()).round() as i32;
        if x >= 0 && (x as usize)<w && y>=0 && (y as usize)<h { buf[y as usize*w+x as usize]=color; } }
}

struct FpsCounter { last_time: Instant, frame_count: u64, fps: f64 }
impl FpsCounter {
    fn new() -> Self { Self { last_time: Instant::now(), frame_count: 0, fps: 0.0 } }
    fn tick(&mut self) { self.frame_count += 1; let e = self.last_time.elapsed().as_secs_f64(); if e >= 1.0 { self.fps = self.frame_count as f64/e; self.frame_count=0; self.last_time=Instant::now(); } }
    fn fps(&self) -> f64 { self.fps }
}
