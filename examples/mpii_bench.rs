//! Benchmark Saccade's ridge regression on the MPIIGaze dataset.
//!
//! This gives an honest accuracy number comparable to published literature,
//! unlike click-based validation (see survey §Evaluation and EXPERIMENTS.md §E12).
//!
//! # Dataset
//!
//! MPIIGaze (Zhang et al. 2015/2017):
//!   15 subjects, laptop webcams, 213k images, programmatic ground truth.
//!
//! # Download
//!
//! ```sh
//! curl -L -o MPIIGaze.tar.gz \
//!   http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
//! tar xzf MPIIGaze.tar.gz
//! ```
//!
//! # Usage
//!
//! ```sh
//! cargo run --release --example mpii_bench -- ./MPIIGaze
//! ```
//!
//! # Label format (MPIIGaze/Label/p00.label)
//!
//! Each line (space-separated):
//!   image_rel_path  face_x face_y face_w face_h
//!   left_eye_cx left_eye_cy  right_eye_cx right_eye_cy
//!   gaze_vec_x gaze_vec_y gaze_vec_z        ← 3-D unit vector (camera frame)
//!   head_vec_x head_vec_y head_vec_z
//!
//! # Protocol
//!
//! Per subject: first N_CALIB frames → calibration, remaining → test.
//! Target = (yaw, pitch) derived from 3-D gaze vector.
//! Error  = arccos(predicted_unit_vec · true_unit_vec) in degrees.
//! Reports mean ± std over all test frames, per subject and overall.

use image::ImageReader;
use saccade::ridge::{self, RidgeRegressor, BOTH_EYES_FEAT_LEN};

/// Number of frames per subject used for calibration.
const N_CALIB: usize = 200;

/// Feature vector length: two eyes (120-D) + 6 head-pose proxy features.
const FEAT_LEN: usize = BOTH_EYES_FEAT_LEN + 6;

/// Eye patch size in pixels, proportional to face width.
const EYE_PATCH_FRACTION_W: f32 = 0.28;
const EYE_PATCH_FRACTION_H: f32 = 0.13;

/// Lambda candidates for auto-tuning.
const LAMBDA_CANDIDATES: &[f64] = &[1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6];

// ─── Data types ─────────────────────────────────────────────────────────────

struct Sample {
    image_path: String,
    face_x: f32,
    face_y: f32,
    face_w: f32,
    face_h: f32,
    left_eye_cx: f32,
    left_eye_cy: f32,
    right_eye_cx: f32,
    right_eye_cy: f32,
    /// 3-D unit gaze vector in camera frame (x right, y down, z into screen).
    gaze: [f64; 3],
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let dataset_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "MPIIGaze".to_string());

    println!("MPIIGaze benchmark — dataset: {dataset_root}");
    println!("Protocol: first {N_CALIB} frames/subject = calibration, rest = test");
    println!("Features: {FEAT_LEN}-D (CLAHE eye patches + head-pose proxy)");
    println!();

    let mut all_errors: Vec<f64> = Vec::new();
    let mut subjects_done = 0usize;

    for subject_id in 0..15 {
        let subject = format!("p{subject_id:02}");
        let label_path = format!("{dataset_root}/Label/{subject}.label");

        if !std::path::Path::new(&label_path).exists() {
            eprintln!("  skip {subject}: {label_path} not found");
            continue;
        }

        let samples = match parse_label_file(&label_path, &dataset_root) {
            Ok(s) => s,
            Err(e) => { eprintln!("  skip {subject}: {e}"); continue; }
        };

        if samples.len() < N_CALIB + 10 {
            eprintln!("  skip {subject}: only {} samples (need >{})", samples.len(), N_CALIB + 10);
            continue;
        }

        print!("  {subject}: extracting features ({} frames)...", samples.len());
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let extracted: Vec<Option<(Vec<f32>, f64, f64)>> = samples.iter()
            .map(|s| extract_features(s))
            .collect();

        let calib_count = extracted.iter().take(N_CALIB)
            .filter(|e| e.is_some()).count();

        let mut reg = RidgeRegressor::new(N_CALIB + 10, 1e4, FEAT_LEN);

        // Calibration phase
        for entry in extracted.iter().take(N_CALIB).flatten() {
            let (feats, yaw, pitch) = entry;
            // Scale to ~pixel magnitude so lambda auto-tuning works in familiar units
            reg.add_sample(feats.clone(), (*yaw * 1000.0) as f32, (*pitch * 1000.0) as f32);
        }

        // Auto-tune lambda via LOO CV
        if let Some(best_lam) = reg.auto_lambda(LAMBDA_CANDIDATES) {
            reg.set_lambda(best_lam);
        }

        // Test phase
        let mut subject_errors: Vec<f64> = Vec::new();
        for (_i, entry) in extracted.iter().enumerate().skip(N_CALIB) {
            let Some((feats, true_yaw, true_pitch)) = entry else { continue };
            let Some((py_scaled, pp_scaled)) = reg.predict(feats) else { continue };

            let pred_yaw   = py_scaled as f64 / 1000.0;
            let pred_pitch = pp_scaled as f64 / 1000.0;

            // Angular error: compare unit vectors
            let true_vec  = gaze_from_angles(*true_yaw,  *true_pitch);
            let pred_vec  = gaze_from_angles(pred_yaw,   pred_pitch);
            let dot = (true_vec[0] * pred_vec[0]
                     + true_vec[1] * pred_vec[1]
                     + true_vec[2] * pred_vec[2]).clamp(-1.0, 1.0);
            subject_errors.push(dot.acos().to_degrees());
        }

        if subject_errors.is_empty() {
            println!(" no test samples");
            continue;
        }

        let mean_e = subject_errors.iter().sum::<f64>() / subject_errors.len() as f64;
        let std_e  = {
            let v = subject_errors.iter().map(|e| (e - mean_e).powi(2)).sum::<f64>()
                    / subject_errors.len() as f64;
            v.sqrt()
        };
        println!(" {mean_e:.2}° ± {std_e:.2}°  (calib={calib_count}, test={})",
            subject_errors.len());

        all_errors.extend(subject_errors);
        subjects_done += 1;
    }

    if all_errors.is_empty() {
        eprintln!("\nNo results. Check dataset path: {dataset_root}/Label/p00.label");
        eprintln!("Download: curl -L -o MPIIGaze.tar.gz \\");
        eprintln!("  http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz");
        std::process::exit(1);
    }

    let mean   = all_errors.iter().sum::<f64>() / all_errors.len() as f64;
    let std    = {
        let v = all_errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>()
                / all_errors.len() as f64;
        v.sqrt()
    };
    let mut sorted = all_errors.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    println!();
    println!("=== MPIIGaze result ({subjects_done}/15 subjects) ===");
    println!("  Mean error:   {mean:.2}° ± {std:.2}°");
    println!("  Median error: {median:.2}°");
    println!("  Samples:      {}", all_errors.len());
    println!();
    println!("Literature reference (MPIIGaze protocol, no calibration):");
    println!("  WebGazer.js   ~4.0°  (click-calibrated, different protocol)");
    println!("  L2CS-Net       3.92° (no calibration, cross-subject)");
    println!("  FAZE           3.18° (9-point calibration, cross-subject)");
    println!("  GazeTR-Hybrid  3.43° (no calibration, cross-subject)");
}

// ─── Label parsing ───────────────────────────────────────────────────────────

fn parse_label_file(label_path: &str, dataset_root: &str) -> Result<Vec<Sample>, String> {
    let text = std::fs::read_to_string(label_path)
        .map_err(|e| format!("read error: {e}"))?;

    let mut samples = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 15 {
            eprintln!("    line {lineno}: expected ≥15 cols, got {}", cols.len());
            continue;
        }

        let parse_f = |s: &str| s.parse::<f32>().ok();
        let parse_d = |s: &str| s.parse::<f64>().ok();

        let Some(face_x) = parse_f(cols[1]) else { continue };
        let Some(face_y) = parse_f(cols[2]) else { continue };
        let Some(face_w) = parse_f(cols[3]) else { continue };
        let Some(face_h) = parse_f(cols[4]) else { continue };
        let Some(lx)     = parse_f(cols[5]) else { continue };
        let Some(ly)     = parse_f(cols[6]) else { continue };
        let Some(rx)     = parse_f(cols[7]) else { continue };
        let Some(ry)     = parse_f(cols[8]) else { continue };
        let Some(gx)     = parse_d(cols[9])  else { continue };
        let Some(gy)     = parse_d(cols[10]) else { continue };
        let Some(gz)     = parse_d(cols[11]) else { continue };

        // Image path: the label stores a relative path like "./p00/day01/0000001.jpg"
        let rel = cols[0].trim_start_matches("./");
        let image_path = format!("{dataset_root}/Data/{rel}");

        samples.push(Sample {
            image_path,
            face_x, face_y, face_w, face_h,
            left_eye_cx: lx, left_eye_cy: ly,
            right_eye_cx: rx, right_eye_cy: ry,
            gaze: [gx, gy, gz],
        });
    }

    Ok(samples)
}

// ─── Feature extraction ──────────────────────────────────────────────────────

/// Extract features from one sample. Returns (features, yaw_rad, pitch_rad) or None.
fn extract_features(s: &Sample) -> Option<(Vec<f32>, f64, f64)> {
    let img = ImageReader::open(&s.image_path).ok()?.decode().ok()?.into_rgb8();
    let (iw, ih) = (img.width() as f32, img.height() as f32);

    // --- Eye patch extraction using provided eye centres ---
    let right_patch = crop_eye_patch(&img, s.right_eye_cx, s.right_eye_cy, s.face_w, iw, ih)?;
    let left_patch  = crop_eye_patch(&img, s.left_eye_cx,  s.left_eye_cy,  s.face_w, iw, ih)?;

    let r_feat = ridge::extract_eye_features(&right_patch.0, right_patch.1, right_patch.2);
    let l_feat = ridge::extract_eye_features(&left_patch.0,  left_patch.1,  left_patch.2);

    // --- Head-pose proxy features (6) ---
    // Same 6 features as webgazer.rs: face centre, size, roll proxy, inter-eye distance.
    let face_cx_n = (s.face_x + s.face_w / 2.0) / iw * 100.0;
    let face_cy_n = (s.face_y + s.face_h / 2.0) / ih * 100.0;
    let face_w_n  = s.face_w / iw * 100.0;
    let face_h_n  = s.face_h / ih * 100.0;
    let dx = s.left_eye_cx - s.right_eye_cx;
    let dy = s.left_eye_cy - s.right_eye_cy;
    let head_roll = dy.atan2(dx) * 100.0;
    let inter_eye = (dx * dx + dy * dy).sqrt();

    let mut feats = Vec::with_capacity(FEAT_LEN);
    feats.extend_from_slice(&r_feat);
    feats.extend_from_slice(&l_feat);
    feats.push(face_cx_n);
    feats.push(face_cy_n);
    feats.push(face_w_n);
    feats.push(face_h_n);
    feats.push(head_roll);
    feats.push(inter_eye);

    // --- Gaze angles from 3-D vector ---
    // MPIIGaze convention: x=right, y=down, z into screen.
    // yaw   = atan2(gx, gz)  (positive = gaze right)
    // pitch = asin(-gy)      (positive = gaze up, flipping y-down)
    let [gx, gy, gz] = s.gaze;
    let norm = (gx * gx + gy * gy + gz * gz).sqrt();
    if norm < 1e-6 { return None; }
    let yaw   = (gx / norm).atan2(gz / norm);
    let pitch = ((-gy) / norm).asin();

    Some((feats, yaw, pitch))
}

/// Crop an eye patch centred at (cx, cy), sized relative to face_w.
/// Returns (rgb_bytes, width, height) or None if out of bounds.
fn crop_eye_patch(
    img: &image::RgbImage,
    cx: f32, cy: f32,
    face_w: f32,
    iw: f32, ih: f32,
) -> Option<(Vec<u8>, usize, usize)> {
    let pw = (face_w * EYE_PATCH_FRACTION_W).max(8.0) as u32;
    let ph = (face_w * EYE_PATCH_FRACTION_H).max(6.0) as u32;
    let x0 = (cx - pw as f32 / 2.0).max(0.0) as u32;
    let y0 = (cy - ph as f32 / 2.0).max(0.0) as u32;
    if x0 + pw > iw as u32 || y0 + ph > ih as u32 { return None; }

    let sub = image::imageops::crop_imm(img, x0, y0, pw, ph).to_image();
    let rgb: Vec<u8> = sub.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
    Some((rgb, pw as usize, ph as usize))
}

// ─── Geometry helpers ────────────────────────────────────────────────────────

/// Convert (yaw, pitch) in radians to a unit 3-D gaze vector.
/// Convention: x=right, y=down, z=into-screen (MPIIGaze camera frame).
fn gaze_from_angles(yaw: f64, pitch: f64) -> [f64; 3] {
    let x = yaw.sin() * pitch.cos();
    let y = -(pitch.sin());          // positive pitch = up → negative y in y-down frame
    let z = yaw.cos() * pitch.cos();
    let norm = (x * x + y * y + z * z).sqrt().max(1e-10);
    [x / norm, y / norm, z / norm]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaze_from_angles_forward() {
        // yaw=0, pitch=0 → straight ahead (0, 0, 1)
        let v = gaze_from_angles(0.0, 0.0);
        assert!((v[0]).abs() < 1e-9);
        assert!((v[1]).abs() < 1e-9);
        assert!((v[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn angular_error_identical_vectors() {
        let v = gaze_from_angles(0.2, -0.1);
        let dot = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        let err = dot.clamp(-1.0, 1.0).acos().to_degrees();
        assert!(err < 1e-6, "self-angle should be 0°, got {err}");
    }
}
