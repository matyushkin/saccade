//! Quick benchmark for Timm & Barth at different resolutions.

use saccade::frame::GrayFrame;
use saccade::timm::{detect_center, TimmConfig};
use std::time::Instant;

fn make_eye(w: u32, h: u32) -> Vec<u8> {
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let r = w.min(h) as f64 / 4.0;
    let mut data = vec![180u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let d = (((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt()) / r;
            if d < 1.0 {
                data[(y * w + x) as usize] = (30.0 + 150.0 * d) as u8;
            }
        }
    }
    data
}

fn bench(label: &str, w: u32, h: u32, iterations: u32) {
    let data = make_eye(w, h);
    let frame = GrayFrame::new(w, h, &data);
    let config = TimmConfig::default();

    // Warmup
    detect_center(&frame, &config);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = detect_center(&frame, &config);
    }
    let elapsed = start.elapsed();
    let per_frame = elapsed / iterations;
    println!("{label:>20}: {per_frame:>10.2?} / frame ({iterations} iterations)");
}

fn main() {
    println!("Timm & Barth benchmark (release mode)\n");
    bench("40×30 (ROI fast)", 40, 30, 1000);
    bench("80×60 (ROI medium)", 80, 60, 500);
    bench("160×120 (low-res)", 160, 120, 100);
    bench("320×240 (high-res)", 320, 240, 20);
}
