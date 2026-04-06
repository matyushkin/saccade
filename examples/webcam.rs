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
use saccade::frame::GrayFrame;
use saccade::pure::PureConfig;
use saccade::tracker::{Tracker, TrackerConfig, TrackingMode};
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

    // Eye trackers
    let tracker_config = TrackerConfig {
        pure: PureConfig {
            canny_low: 10.0,
            canny_high: 30.0,
            ..PureConfig::default()
        },
        ..TrackerConfig::default()
    };
    let mut left_tracker = Tracker::new(tracker_config.clone());
    let mut right_tracker = Tracker::new(tracker_config);

    // Smoothers — low alpha = more smoothing
    let mut face_smooth = SmoothRect::new(0.3);
    let mut left_eye_smooth = SmoothRect::new(0.3);
    let mut right_eye_smooth = SmoothRect::new(0.3);
    let mut left_pupil_smooth = SmoothPoint::new(0.2);
    let mut right_pupil_smooth = SmoothPoint::new(0.2);

    let mut frame_buf = vec![0u32; cam_w * cam_h];
    let mut fps_counter = FpsCounter::new();
    let mut no_face_count = 0u32;

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

        if let Some(face) = best_face {
            no_face_count = 0;
            let bbox = face.bbox();
            let fx = bbox.x().max(0) as f64;
            let fy = bbox.y().max(0) as f64;
            let fw = bbox.width() as f64;
            let fh = bbox.height() as f64;

            // Smooth face bbox
            face_smooth.update(fx, fy, fw, fh);
            let (sfx, sfy, sfw, sfh) = face_smooth.get();

            draw_rect(&mut frame_buf, cam_w, cam_h, sfx, sfy, sfw, sfh, 0xFFFF00);

            // Eye regions from smoothed face
            let eye_y = sfy + sfh * 28 / 100;
            let eye_h = sfh * 25 / 100;
            let left_eye_x = sfx + sfw * 8 / 100;
            let right_eye_x = sfx + sfw * 52 / 100;
            let eye_w = sfw * 40 / 100;

            // Smooth eye ROIs
            left_eye_smooth.update(left_eye_x as f64, eye_y as f64, eye_w as f64, eye_h as f64);
            right_eye_smooth.update(right_eye_x as f64, eye_y as f64, eye_w as f64, eye_h as f64);

            let (lx, ly, lw, lh) = left_eye_smooth.get();
            let (rx, ry, rw, rh) = right_eye_smooth.get();

            // Track left eye
            if let Some(result) = track_eye(&gray, cam_w, cam_h, lx, ly, lw, lh, &mut left_tracker) {
                draw_rect(&mut frame_buf, cam_w, cam_h, lx, ly, lw, lh, 0x00FF00);
                if let Some(pupil) = result.pupil {
                    let abs_x = lx as f64 + pupil.cx;
                    let abs_y = ly as f64 + pupil.cy;
                    left_pupil_smooth.update(abs_x, abs_y);
                    let (px, py) = left_pupil_smooth.get();
                    draw_crosshair(&mut frame_buf, cam_w, cam_h, px, py, 0xFF0000, result.mode);
                }
            }

            // Track right eye
            if let Some(result) = track_eye(&gray, cam_w, cam_h, rx, ry, rw, rh, &mut right_tracker) {
                draw_rect(&mut frame_buf, cam_w, cam_h, rx, ry, rw, rh, 0x00FF00);
                if let Some(pupil) = result.pupil {
                    let abs_x = rx as f64 + pupil.cx;
                    let abs_y = ry as f64 + pupil.cy;
                    right_pupil_smooth.update(abs_x, abs_y);
                    let (px, py) = right_pupil_smooth.get();
                    draw_crosshair(&mut frame_buf, cam_w, cam_h, px, py, 0x00FFFF, result.mode);
                }
            }
        } else {
            no_face_count += 1;
            if no_face_count > 30 {
                // Lost face for >1 second — reset
                left_tracker.reset();
                right_tracker.reset();
                left_pupil_smooth = SmoothPoint::new(0.2);
                right_pupil_smooth = SmoothPoint::new(0.2);
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
            print!("\rFPS: {fps:.1} | Frame: {frame_ms}ms | Faces: {}    ", faces.len());
        }
    }

    println!("\nDone.");
}

struct TrackResult {
    pupil: Option<saccade::ellipse::Ellipse>,
    mode: TrackingMode,
}

fn track_eye(
    gray: &[u8],
    img_w: usize,
    img_h: usize,
    roi_x: usize,
    roi_y: usize,
    roi_w: usize,
    roi_h: usize,
    tracker: &mut Tracker,
) -> Option<TrackResult> {
    if roi_x + roi_w > img_w || roi_y + roi_h > img_h || roi_w < 20 || roi_h < 15 {
        return None;
    }

    let mut eye_data = vec![0u8; roi_w * roi_h];
    for y in 0..roi_h {
        let src_start = (roi_y + y) * img_w + roi_x;
        let dst_start = y * roi_w;
        eye_data[dst_start..dst_start + roi_w]
            .copy_from_slice(&gray[src_start..src_start + roi_w]);
    }

    let frame = GrayFrame::new(roi_w as u32, roi_h as u32, &eye_data);
    let result = tracker.track(&frame);

    Some(TrackResult {
        pupil: result.pupil,
        mode: result.mode,
    })
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

fn draw_crosshair(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, color: u32, mode: TrackingMode) {
    let size: i32 = match mode {
        TrackingMode::Fast => 4,
        TrackingMode::Precise => 6,
        TrackingMode::FullScan => 8,
    };
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
