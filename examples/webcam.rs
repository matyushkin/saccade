//! Real-time eye tracking from webcam.
//!
//! Usage:
//!   cargo run --release --features demo --example webcam
//!
//! Requires: webcam access permission on macOS.
//! Downloads face detection model on first run.

use minifb::{Key, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use saccade::blink::{BlinkDetector, EyeState};
use saccade::classify::{EyeEvent, IVTClassifier};
use saccade::frame::GrayFrame;
use saccade::timm::{self, TimmConfig};
use std::time::Instant;

const MODEL_PATH: &str = "seeta_fd_frontal_v1.0.bin";
const MODEL_URL: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";

/// Exponential moving average smoother for a rectangle (x, y, w, h).
struct SmoothRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    alpha: f64,
    initialized: bool,
}

impl SmoothRect {
    fn new(alpha: f64) -> Self {
        Self { x: 0.0, y: 0.0, w: 0.0, h: 0.0, alpha, initialized: false }
    }

    fn update(&mut self, x: f64, y: f64, w: f64, h: f64) {
        if !self.initialized {
            self.x = x;
            self.y = y;
            self.w = w;
            self.h = h;
            self.initialized = true;
        } else {
            let a = self.alpha;
            self.x = a * x + (1.0 - a) * self.x;
            self.y = a * y + (1.0 - a) * self.y;
            self.w = a * w + (1.0 - a) * self.w;
            self.h = a * h + (1.0 - a) * self.h;
        }
    }

    fn get(&self) -> (usize, usize, usize, usize) {
        (self.x.round() as usize, self.y.round() as usize,
         self.w.round().max(1.0) as usize, self.h.round().max(1.0) as usize)
    }
}

/// Exponential moving average smoother for a 2D point.
struct SmoothPoint {
    x: f64,
    y: f64,
    alpha: f64,
    initialized: bool,
}

impl SmoothPoint {
    fn new(alpha: f64) -> Self {
        Self { x: 0.0, y: 0.0, alpha, initialized: false }
    }

    fn update(&mut self, x: f64, y: f64) {
        if !self.initialized {
            self.x = x;
            self.y = y;
            self.initialized = true;
        } else {
            let a = self.alpha;
            self.x = a * x + (1.0 - a) * self.x;
            self.y = a * y + (1.0 - a) * self.y;
        }
    }

    fn get(&self) -> (usize, usize) {
        (self.x.round() as usize, self.y.round() as usize)
    }
}

fn main() {
    // Ensure face detection model exists
    if !std::path::Path::new(MODEL_PATH).exists() {
        println!("Downloading face detection model...");
        let status = std::process::Command::new("curl")
            .args(["-L", "-o", MODEL_PATH, MODEL_URL])
            .status()
            .expect("Failed to run curl");
        if !status.success() {
            eprintln!("Failed to download model. Please download manually:");
            eprintln!("  curl -L -o {MODEL_PATH} {MODEL_URL}");
            std::process::exit(1);
        }
        println!("Model downloaded.");
    }

    // Initialize face detector
    let mut detector = rustface::create_detector(MODEL_PATH)
        .expect("Failed to create face detector");
    detector.set_min_face_size(80);
    detector.set_score_thresh(2.0);

    // Initialize camera
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(0), format)
        .expect("Failed to open camera");
    camera.open_stream().expect("Failed to open camera stream");

    let cam_res = camera.resolution();
    println!("Camera native: {}×{}", cam_res.width(), cam_res.height());

    let scale_down = (cam_res.width() / 640).max(1) as usize;
    let cam_w = cam_res.width() as usize / scale_down;
    let cam_h = cam_res.height() as usize / scale_down;
    println!("Processing at: {cam_w}×{cam_h} (scale 1/{scale_down})");

    // Initialize display window
    let mut window = Window::new(
        "Saccade — Eye Tracker (ESC to quit)",
        cam_w,
        cam_h,
        WindowOptions::default(),
    )
    .expect("Failed to create window");
    window.set_target_fps(60);

    // Timm & Barth config — tuned for small webcam eye ROIs
    let timm_config = TimmConfig {
        gradient_threshold: 0.2, // lower threshold to catch subtle gradients
        use_weight_map: true,
        weight_blur_sigma: 2.0,
    };

    // Smoothers
    let mut face_smooth = SmoothRect::new(0.25);
    let mut left_eye_smooth = SmoothRect::new(0.25);
    let mut right_eye_smooth = SmoothRect::new(0.25);
    let mut left_pupil_smooth = SmoothPoint::new(0.4); // less lag
    let mut right_pupil_smooth = SmoothPoint::new(0.4);

    // Blink detectors
    let mut left_blink = BlinkDetector::new();
    left_blink.confidence_threshold = 0.06; // dark_ratio below this = eye closed
    let mut right_blink = BlinkDetector::new();
    right_blink.confidence_threshold = 0.06;

    // Smooth openness signal before feeding to blink detector
    let mut left_openness_ema = 0.5f64;
    let mut right_openness_ema = 0.5f64;
    let openness_alpha = 0.4;

    // Per-eye adaptive threshold calibration
    let mut left_baseline_sum = 0.0f64;
    let mut right_baseline_sum = 0.0f64;
    let mut calibration_frames = 0u32;
    let calibration_period = 45; // ~3 seconds at 15fps
    let mut classifier = IVTClassifier::default_params();
    let start_time = Instant::now();

    let mut frame_buf = vec![0u32; cam_w * cam_h];
    let mut fps_counter = FpsCounter::new();
    let mut no_face_count = 0u32;
    let mut face_present = false;

    println!("Running... Press ESC to quit.");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = Instant::now();

        // Capture frame
        let decoded = match camera.frame() {
            Ok(f) => f.decode_image::<RgbFormat>(),
            Err(_) => continue,
        };
        let rgb_image = match decoded {
            Ok(img) => img,
            Err(_) => continue,
        };
        let full_w = rgb_image.width() as usize;
        let full_h = rgb_image.height() as usize;
        let rgb_full = rgb_image.as_raw();

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
                    gray[y * cam_w + x] =
                        (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                }
            }
        }

        // Detect faces
        let image_data = ImageData::new(&gray, cam_w as u32, cam_h as u32);
        let faces = detector.detect(&image_data);

        // RGB → display buffer
        for i in 0..cam_w * cam_h {
            let r = rgb_data[i * 3] as u32;
            let g = rgb_data[i * 3 + 1] as u32;
            let b = rgb_data[i * 3 + 2] as u32;
            frame_buf[i] = (r << 16) | (g << 8) | b;
        }

        // Pick the largest face (most likely the user)
        let best_face = faces.iter().max_by_key(|f| {
            let b = f.bbox();
            b.width() as i32 * b.height() as i32
        });

        let now_ms = start_time.elapsed().as_millis() as u64;

        if let Some(face) = best_face {
            no_face_count = 0;
            face_present = true;
            let bbox = face.bbox();
            let fx = bbox.x().max(0) as f64;
            let fy = bbox.y().max(0) as f64;
            let fw = bbox.width() as f64;
            let fh = bbox.height() as f64;

            // Smooth face bbox
            face_smooth.update(fx, fy, fw, fh);
            let (sfx, sfy, sfw, sfh) = face_smooth.get();

            draw_rect(&mut frame_buf, cam_w, cam_h, sfx, sfy, sfw, sfh, 0xFFFF00);

            // Tighter eye regions — less eyebrow, more focused on eye
            let eye_y = sfy + sfh * 32 / 100;
            let eye_h = sfh * 18 / 100;
            let left_eye_x = sfx + sfw * 10 / 100;
            let right_eye_x = sfx + sfw * 55 / 100;
            let eye_w = sfw * 35 / 100;

            // Smooth eye ROIs
            left_eye_smooth.update(left_eye_x as f64, eye_y as f64, eye_w as f64, eye_h as f64);
            right_eye_smooth.update(right_eye_x as f64, eye_y as f64, eye_w as f64, eye_h as f64);

            let (lx, ly, lw, lh) = left_eye_smooth.get();
            let (rx, ry, rw, rh) = right_eye_smooth.get();

            // Track left eye
            let left_openness = if let Some((cx, cy, dark_ratio)) = detect_eye(&gray, cam_w, cam_h, lx, ly, lw, lh, &timm_config) {
                draw_rect(&mut frame_buf, cam_w, cam_h, lx, ly, lw, lh, 0x00FF00);

                // Smooth the dark_ratio signal
                left_openness_ema = openness_alpha * dark_ratio + (1.0 - openness_alpha) * left_openness_ema;

                if left_openness_ema > 0.06 {
                    left_pupil_smooth.update(cx, cy);
                    let (px, py) = left_pupil_smooth.get();
                    draw_crosshair(&mut frame_buf, cam_w, cam_h, px, py, 0xFF0000);
                    classifier.update(cx, cy, now_ms);
                }
                left_openness_ema
            } else {
                0.0
            };

            // Track right eye
            let right_openness = if let Some((cx, cy, dark_ratio)) = detect_eye(&gray, cam_w, cam_h, rx, ry, rw, rh, &timm_config) {
                draw_rect(&mut frame_buf, cam_w, cam_h, rx, ry, rw, rh, 0x00FF00);

                right_openness_ema = openness_alpha * dark_ratio + (1.0 - openness_alpha) * right_openness_ema;

                if right_openness_ema > 0.06 {
                    right_pupil_smooth.update(cx, cy);
                    let (px, py) = right_pupil_smooth.get();
                    draw_crosshair(&mut frame_buf, cam_w, cam_h, px, py, 0x00FFFF);
                }
                right_openness_ema
            } else {
                0.0
            };

            // Adaptive calibration: learn per-eye baseline during first N frames
            if calibration_frames < calibration_period {
                left_baseline_sum += left_openness;
                right_baseline_sum += right_openness;
                calibration_frames += 1;

                if calibration_frames == calibration_period {
                    let left_base = left_baseline_sum / calibration_period as f64;
                    let right_base = right_baseline_sum / calibration_period as f64;
                    // Threshold = 50% of baseline (closed eye has ~20-30% of open dark ratio)
                    left_blink.confidence_threshold = (left_base * 0.5).max(0.03);
                    right_blink.confidence_threshold = (right_base * 0.5).max(0.03);
                    eprintln!("\nCalibrated: L_base={left_base:.3} thresh={:.3} | R_base={right_base:.3} thresh={:.3}",
                        left_blink.confidence_threshold, right_blink.confidence_threshold);
                }
            }

            // Update blink detectors
            let left_eye_state = left_blink.update(left_openness, now_ms);
            let right_eye_state = right_blink.update(right_openness, now_ms);

            // Draw eye state indicators (colored dots in top-right)
            let state_y = 15;
            let left_color = eye_state_color(left_eye_state);
            let right_color = eye_state_color(right_eye_state);
            draw_filled_circle(&mut frame_buf, cam_w, cam_h, cam_w - 40, state_y, 8, left_color);
            draw_filled_circle(&mut frame_buf, cam_w, cam_h, cam_w - 20, state_y, 8, right_color);
        } else {
            no_face_count += 1;
            face_present = false;
            if no_face_count > 30 {
                left_pupil_smooth = SmoothPoint::new(0.4);
                right_pupil_smooth = SmoothPoint::new(0.4);
                left_blink.reset();
                right_blink.reset();
                no_face_count = 0;
            } else if face_smooth.initialized {
                // Use last known face position for a few frames
                let (sfx, sfy, sfw, sfh) = face_smooth.get();
                draw_rect(&mut frame_buf, cam_w, cam_h, sfx, sfy, sfw, sfh, 0x666600);
            }
        }

        // FPS bar
        fps_counter.tick();
        let fps = fps_counter.fps();
        let frame_ms = frame_start.elapsed().as_millis();
        let bar_len = (fps as usize * 3).min(cam_w);
        let bar_color = if fps > 20.0 { 0x00FF00 } else if fps > 10.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar_len {
            for y in 0..4 {
                frame_buf[y * cam_w + x] = bar_color;
            }
        }

        window.update_with_buffer(&frame_buf, cam_w, cam_h).unwrap();

        if fps_counter.frame_count % 30 == 0 {
            let l_state = if face_present { format!("{:?}", left_blink.state()) } else { "NoFace".into() };
            let r_state = if face_present { format!("{:?}", right_blink.state()) } else { "NoFace".into() };
            let blinks = left_blink.blink_count() + right_blink.blink_count();
            let bpm = (left_blink.blinks_per_minute(now_ms, 60_000)
                + right_blink.blinks_per_minute(now_ms, 60_000)) / 2.0;
            let fixations = classifier.events().iter()
                .filter(|e| matches!(e, EyeEvent::Fixation(_))).count();
            let saccades = classifier.events().iter()
                .filter(|e| matches!(e, EyeEvent::Saccade(_))).count();
            print!("\rFPS:{fps:.0} | L:{l_state} R:{r_state} | Blinks:{blinks} ({bpm:.0}/min) | Fix:{fixations} Sac:{saccades}    ");
        }
    }

    println!("\nDone.");
}

/// Detect eye center using Timm & Barth with contrast enhancement.
/// Returns (abs_x, abs_y, openness) in frame coordinates.
/// `openness` is based on pixel standard deviation in the ROI:
/// high std = open eye (dark pupil + bright sclera), low std = closed.
fn detect_eye(
    gray: &[u8],
    img_w: usize,
    img_h: usize,
    roi_x: usize,
    roi_y: usize,
    roi_w: usize,
    roi_h: usize,
    config: &TimmConfig,
) -> Option<(f64, f64, f64)> {
    if roi_x + roi_w > img_w || roi_y + roi_h > img_h || roi_w < 10 || roi_h < 8 {
        return None;
    }

    // Extract eye region
    let mut eye_data = vec![0u8; roi_w * roi_h];
    for y in 0..roi_h {
        let src_start = (roi_y + y) * img_w + roi_x;
        let dst_start = y * roi_w;
        eye_data[dst_start..dst_start + roi_w]
            .copy_from_slice(&gray[src_start..src_start + roi_w]);
    }

    // Compute eye openness using min-region darkness ratio.
    // An open eye has a dark pupil region (bottom 10% of brightness).
    // A closed eye is more uniform (eyelid color).
    let n = eye_data.len();
    let mean = eye_data.iter().map(|&p| p as u32).sum::<u32>() / n as u32;
    // Dark threshold: pixels below 60% of mean brightness
    let dark_thresh = (mean as f64 * 0.6) as u8;
    let dark_count = eye_data.iter().filter(|&&p| p < dark_thresh).count();
    let dark_ratio = dark_count as f64 / n as f64;

    // Contrast stretch
    let min_val = *eye_data.iter().min().unwrap_or(&0);
    let max_val = *eye_data.iter().max().unwrap_or(&255);
    if max_val > min_val + 10 {
        let range = (max_val - min_val) as f32;
        for p in &mut eye_data {
            *p = ((*p as f32 - min_val as f32) / range * 255.0) as u8;
        }
    }

    let frame = GrayFrame::new(roi_w as u32, roi_h as u32, &eye_data);
    let result = timm::detect_center(&frame, config);

    let abs_x = roi_x as f64 + result.x;
    let abs_y = roi_y as f64 + result.y;
    // dark_ratio: ~0.10-0.25 for open eye (pupil visible), ~0.01-0.05 for closed
    Some((abs_x, abs_y, dark_ratio))
}

fn draw_rect(buf: &mut [u32], w: usize, h: usize, x: usize, y: usize, rw: usize, rh: usize, color: u32) {
    for dx in 0..rw {
        let px = x + dx;
        if px < w {
            if y < h { buf[y * w + px] = color; }
            let by = y + rh.saturating_sub(1);
            if by < h { buf[by * w + px] = color; }
        }
    }
    for dy in 0..rh {
        let py = y + dy;
        if py < h {
            if x < w { buf[py * w + x] = color; }
            let bx = x + rw.saturating_sub(1);
            if bx < w { buf[py * w + bx] = color; }
        }
    }
}

fn eye_state_color(state: EyeState) -> u32 {
    match state {
        EyeState::Open => 0x00FF00,    // green
        EyeState::Blinking => 0xFFFF00, // yellow
        EyeState::Closed => 0xFF0000,   // red
    }
}

fn draw_filled_circle(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, r: usize, color: u32) {
    for dy in 0..=r {
        for dx in 0..=r {
            if dx * dx + dy * dy <= r * r {
                for &(sx, sy) in &[
                    (cx + dx, cy + dy), (cx.wrapping_sub(dx), cy + dy),
                    (cx + dx, cy.wrapping_sub(dy)), (cx.wrapping_sub(dx), cy.wrapping_sub(dy)),
                ] {
                    if sx < w && sy < h {
                        buf[sy * w + sx] = color;
                    }
                }
            }
        }
    }
}

fn draw_crosshair(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, color: u32) {
    let size: i32 = 6;
    for d in -size..=size {
        let x = cx as i32 + d;
        if x >= 0 && (x as usize) < w && cy < h {
            buf[cy * w + x as usize] = color;
        }
        let y = cy as i32 + d;
        if y >= 0 && (y as usize) < h && cx < w {
            buf[y as usize * w + cx] = color;
        }
    }
    let r = (size + 2) as f64;
    for i in 0..48 {
        let t = 2.0 * std::f64::consts::PI * i as f64 / 48.0;
        let x = (cx as f64 + r * t.cos()).round() as i32;
        let y = (cy as f64 + r * t.sin()).round() as i32;
        if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
            buf[y as usize * w + x as usize] = color;
        }
    }
}

struct FpsCounter {
    last_time: Instant,
    frame_count: u64,
    fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self { last_time: Instant::now(), frame_count: 0, fps: 0.0 }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let elapsed = self.last_time.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f64 / elapsed;
            self.frame_count = 0;
            self.last_time = Instant::now();
        }
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}
