//! Diagnostic: capture 10 seconds of dark_ratio values to understand the signal.
//! Prints CSV to stdout. Keep eyes open first 3 sec, then close LEFT eye,
//! then open, then close RIGHT eye.
//!
//! cargo run --release --features demo --example diag

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use std::time::Instant;

const MODEL_PATH: &str = "seeta_fd_frontal_v1.0.bin";

fn main() {
    let mut detector = rustface::create_detector(MODEL_PATH)
        .expect("Failed to load model. Run webcam example first to download it.");
    detector.set_min_face_size(80);
    detector.set_score_thresh(2.0);

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(0), format).expect("Failed to open camera");
    camera.open_stream().expect("Failed to open stream");

    let cam_res = camera.resolution();
    let scale_down = (cam_res.width() / 640).max(1) as usize;
    let cam_w = cam_res.width() as usize / scale_down;
    let cam_h = cam_res.height() as usize / scale_down;

    eprintln!("Camera: {}x{} → {}x{}", cam_res.width(), cam_res.height(), cam_w, cam_h);
    eprintln!("Keep eyes OPEN for 3 sec, then close LEFT eye for 3 sec, then open, then close RIGHT for 3 sec.");
    eprintln!();

    println!("time_ms,left_dark_ratio,right_dark_ratio,left_std,right_std,left_min_ratio,right_min_ratio");

    let start = Instant::now();
    let duration_sec = 12;

    while start.elapsed().as_secs() < duration_sec {
        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(img) => img, Err(_) => continue },
            Err(_) => continue,
        };

        let full_w = decoded.width() as usize;
        let full_h = decoded.height() as usize;
        let rgb_full = decoded.as_raw();

        let mut gray = vec![0u8; cam_w * cam_h];
        for y in 0..cam_h {
            for x in 0..cam_w {
                let sx = x * scale_down;
                let sy = y * scale_down;
                if sx < full_w && sy < full_h {
                    let si = (sy * full_w + sx) * 3;
                    let (r, g, b) = (rgb_full[si], rgb_full[si+1], rgb_full[si+2]);
                    gray[y * cam_w + x] = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                }
            }
        }

        let image_data = ImageData::new(&gray, cam_w as u32, cam_h as u32);
        let faces = detector.detect(&image_data);

        let best_face = faces.iter().max_by_key(|f| {
            let b = f.bbox();
            b.width() as i32 * b.height() as i32
        });

        if let Some(face) = best_face {
            let bbox = face.bbox();
            let fx = bbox.x().max(0) as usize;
            let fy = bbox.y().max(0) as usize;
            let fw = bbox.width() as usize;
            let fh = bbox.height() as usize;

            let eye_y = fy + fh * 32 / 100;
            let eye_h = fh * 18 / 100;
            let left_eye_x = fx + fw * 10 / 100;
            let right_eye_x = fx + fw * 55 / 100;
            let eye_w = fw * 35 / 100;

            let l = analyze_roi(&gray, cam_w, cam_h, left_eye_x, eye_y, eye_w, eye_h);
            let r = analyze_roi(&gray, cam_w, cam_h, right_eye_x, eye_y, eye_w, eye_h);

            if let (Some(l), Some(r)) = (l, r) {
                let t = start.elapsed().as_millis();
                println!("{t},{:.4},{:.4},{:.1},{:.1},{:.3},{:.3}",
                    l.dark_ratio, r.dark_ratio, l.std_dev, r.std_dev, l.min_ratio, r.min_ratio);
            }
        }
    }
    eprintln!("Done. Analyze the CSV output.");
}

struct RoiStats {
    dark_ratio: f64,
    std_dev: f64,
    min_ratio: f64, // ratio of min_val / mean — lower = darker pupil present
}

fn analyze_roi(gray: &[u8], img_w: usize, img_h: usize, x: usize, y: usize, w: usize, h: usize) -> Option<RoiStats> {
    if x + w > img_w || y + h > img_h || w < 5 || h < 5 { return None; }

    let mut pixels = Vec::with_capacity(w * h);
    for row in y..y+h {
        for col in x..x+w {
            pixels.push(gray[row * img_w + col]);
        }
    }

    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let variance = pixels.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let dark_thresh = (mean * 0.6) as u8;
    let dark_count = pixels.iter().filter(|&&p| p < dark_thresh).count();
    let dark_ratio = dark_count as f64 / n;

    // Min region: average of darkest 5% pixels / mean
    let mut sorted = pixels.clone();
    sorted.sort();
    let bottom_n = (n * 0.05).max(1.0) as usize;
    let min_avg = sorted[..bottom_n].iter().map(|&p| p as f64).sum::<f64>() / bottom_n as f64;
    let min_ratio = if mean > 1.0 { min_avg / mean } else { 1.0 };

    Some(RoiStats { dark_ratio, std_dev, min_ratio })
}
