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
//! # Download & Preprocess
//!
//! ```sh
//! curl -L -o MPIIGaze.tar.gz \
//!   https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
//! tar xzf MPIIGaze.tar.gz
//! python3 tools/preprocess_mpii.py ./MPIIGaze ./MPIIGaze_proc
//! ```
//!
//! # Usage
//!
//! ```sh
//! cargo run --release --example mpii_bench -- ./MPIIGaze_proc
//! ```
//!
//! # Preprocessed format (MPIIGaze_proc/p00/labels.tsv)
//!
//! Header: idx  gx  gy  gz  head0  head1  head2
//! Images: {idx:07}_left.png and {idx:07}_right.png  (36×60 grayscale)
//!
//! Gaze vector is a 3-D unit vector in MPIIGaze normalized camera frame.
//! x=right, y=down, z=toward camera.
//! yaw   = atan2(-gx, -gz)   pitch = asin(-gy)
//!
//! # Protocol
//!
//! Per subject: first N_CALIB frames → calibration, remaining → test.
//! Target = (yaw, pitch) in radians, scaled ×1000 for ridge regression.
//! Error  = arccos(predicted_unit_vec · true_unit_vec) in degrees.
//! Reports mean ± std over all test frames, per subject and overall.

use image::ImageReader;
use saccade::ridge::{self, RidgeRegressor, BOTH_EYES_FEAT_LEN};

/// Number of frames per subject used for calibration.
const N_CALIB: usize = 200;

/// Feature vector length: two eyes (120-D) + 3 head-pose proxy features.
const FEAT_LEN: usize = BOTH_EYES_FEAT_LEN + 3;

/// Lambda candidates for auto-tuning.
const LAMBDA_CANDIDATES: &[f64] = &[1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6];

// ─── Data types ─────────────────────────────────────────────────────────────

struct Sample {
    left_path:  String,
    right_path: String,
    /// 3-D unit gaze vector in MPIIGaze normalized camera frame.
    gaze: [f64; 3],
    head: [f32; 3],
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let proc_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "MPIIGaze_proc".to_string());

    println!("MPIIGaze benchmark — preprocessed root: {proc_root}");
    println!("Protocol: first {N_CALIB} frames/subject = calibration, rest = test");
    println!("Features: {FEAT_LEN}-D (CLAHE eye patches + 3 head-pose)");
    println!();

    let mut all_errors: Vec<f64> = Vec::new();
    let mut subjects_done = 0usize;

    for subject_id in 0..15 {
        let subject = format!("p{subject_id:02}");
        let label_path = format!("{proc_root}/{subject}/labels.tsv");

        if !std::path::Path::new(&label_path).exists() {
            eprintln!("  skip {subject}: {label_path} not found");
            continue;
        }

        let samples = match parse_labels(&label_path, &format!("{proc_root}/{subject}")) {
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
            .map(extract_features)
            .collect();

        let calib_count = extracted.iter().take(N_CALIB)
            .filter(|e| e.is_some()).count();

        let mut reg = RidgeRegressor::new(N_CALIB + 10, 1e4, FEAT_LEN);

        // Calibration phase
        for entry in extracted.iter().take(N_CALIB).flatten() {
            let (feats, yaw, pitch) = entry;
            // Scale ×1000 so angles (≈ ±0.5 rad) are in a range where lambda auto-tuning works well
            reg.add_sample(feats.clone(), (*yaw * 1000.0) as f32, (*pitch * 1000.0) as f32);
        }

        // Auto-tune lambda via LOO CV
        if let Some(best_lam) = reg.auto_lambda(LAMBDA_CANDIDATES) {
            reg.set_lambda(best_lam);
        }

        // Test phase
        let mut subject_errors: Vec<f64> = Vec::new();
        for entry in extracted.iter().skip(N_CALIB).flatten() {
            let (feats, true_yaw, true_pitch) = entry;
            let Some((py_scaled, pp_scaled)) = reg.predict(feats) else { continue };

            let pred_yaw   = py_scaled as f64 / 1000.0;
            let pred_pitch = pp_scaled as f64 / 1000.0;

            // Angular error: both true and pred as unit vectors
            let true_vec = gaze_from_angles(*true_yaw, *true_pitch);
            let pred_vec = gaze_from_angles(pred_yaw,  pred_pitch);
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
        eprintln!("\nNo results. Did you run the preprocessor?");
        eprintln!("  python3 tools/preprocess_mpii.py ./MPIIGaze ./MPIIGaze_proc");
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

fn parse_labels(label_path: &str, subject_dir: &str) -> Result<Vec<Sample>, String> {
    let text = std::fs::read_to_string(label_path)
        .map_err(|e| format!("read error: {e}"))?;

    let mut samples = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 { continue; }  // skip header
        let line = line.trim();
        if line.is_empty() { continue; }

        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 7 {
            eprintln!("  line {lineno}: expected 7 cols, got {}", cols.len());
            continue;
        }

        let parse_d = |s: &str| s.parse::<f64>().ok();
        let parse_f = |s: &str| s.parse::<f32>().ok();

        let Some(idx) = cols[0].parse::<u64>().ok() else { continue };
        let Some(gx)  = parse_d(cols[1])            else { continue };
        let Some(gy)  = parse_d(cols[2])            else { continue };
        let Some(gz)  = parse_d(cols[3])            else { continue };
        let Some(h0)  = parse_f(cols[4])            else { continue };
        let Some(h1)  = parse_f(cols[5])            else { continue };
        let Some(h2)  = parse_f(cols[6])            else { continue };

        samples.push(Sample {
            left_path:  format!("{subject_dir}/{idx:07}_left.png"),
            right_path: format!("{subject_dir}/{idx:07}_right.png"),
            gaze: [gx, gy, gz],
            head: [h0, h1, h2],
        });
    }

    Ok(samples)
}

// ─── Feature extraction ──────────────────────────────────────────────────────

/// Extract features from one sample. Returns (features, yaw_rad, pitch_rad) or None.
fn extract_features(s: &Sample) -> Option<(Vec<f32>, f64, f64)> {
    let load_gray = |path: &str| -> Option<(Vec<u8>, usize, usize)> {
        let img = ImageReader::open(path).ok()?.decode().ok()?.into_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Some((img.into_raw(), w, h))
    };

    let (left_gray,  lw, lh) = load_gray(&s.left_path)?;
    let (right_gray, rw, rh) = load_gray(&s.right_path)?;

    // Patches are single-channel (grayscale PNG from preprocessor)
    let l_feat = ridge::extract_eye_features_gray(&left_gray,  lw, lh);
    let r_feat = ridge::extract_eye_features_gray(&right_gray, rw, rh);

    // Convert 3-D unit gaze vector → yaw/pitch angles.
    // MPIIGaze convention: x=right, y=down, z=toward camera.
    // yaw   = atan2(-gx, -gz)  (positive = looking left from subject's POV)
    // pitch = asin(-gy)        (positive = looking up)
    let [gx, gy, gz] = s.gaze;
    let norm = (gx*gx + gy*gy + gz*gz).sqrt();
    if norm < 1e-6 { return None; }
    let yaw   = (-gx / norm).atan2(-gz / norm);
    let pitch = ((-gy) / norm).asin();

    let mut feats = Vec::with_capacity(FEAT_LEN);
    feats.extend_from_slice(&l_feat);
    feats.extend_from_slice(&r_feat);
    feats.push(s.head[0]);
    feats.push(s.head[1]);
    feats.push(s.head[2]);

    Some((feats, yaw, pitch))
}

// ─── Geometry helpers ────────────────────────────────────────────────────────

/// Convert (yaw, pitch) in radians to a unit 3-D gaze vector.
/// Convention: x=right, y=down, z=into-screen.
fn gaze_from_angles(yaw: f64, pitch: f64) -> [f64; 3] {
    let x =  yaw.sin() * pitch.cos();
    let y = -(pitch.sin());
    let z =  yaw.cos() * pitch.cos();
    let norm = (x * x + y * y + z * z).sqrt().max(1e-10);
    [x / norm, y / norm, z / norm]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaze_from_angles_forward() {
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
