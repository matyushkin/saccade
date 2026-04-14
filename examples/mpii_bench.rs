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
use saccade::ridge::{self, RidgeRegressor};

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
    // Parse args: [proc_root] [--patch WxH] [--n-calib N] [--flip-right] [--gradient]
    let mut proc_root = "MPIIGaze_proc".to_string();
    let mut patch_w: usize = 20;
    let mut patch_h: usize = 12;
    let mut n_calib: usize = 200;
    let mut flip_right = false;
    let mut use_gradient = false;
    let mut uniform_calib = false;
    let mut clahe_tx: usize = 3;
    let mut clahe_ty: usize = 3;
    let mut clahe_clip: f32 = 4.0;
    let mut normalize_feats = false;
    let mut no_head_pose = false;
    let mut separate_eyes = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--patch" => {
                let v = args.next().expect("--patch WxH");
                let parts: Vec<&str> = v.split('x').collect();
                patch_w = parts[0].parse().expect("patch W");
                patch_h = parts[1].parse().expect("patch H");
            }
            "--n-calib" => {
                n_calib = args.next().expect("--n-calib N").parse().expect("N");
            }
            "--clahe-tiles" => {
                let v = args.next().expect("--clahe-tiles TxT");
                let parts: Vec<&str> = v.split('x').collect();
                clahe_tx = parts[0].parse().expect("tiles X");
                clahe_ty = parts[1].parse().expect("tiles Y");
            }
            "--clahe-clip" => {
                clahe_clip = args.next().expect("--clahe-clip C").parse().expect("C");
            }
            "--flip-right"         => flip_right = true,
            "--gradient"           => use_gradient = true,
            "--uniform-calib"      => uniform_calib = true,
            "--normalize-features" => normalize_feats = true,
            "--no-head-pose"       => no_head_pose = true,
            "--separate-eyes"      => separate_eyes = true,
            s if !s.starts_with('-') => proc_root = s.to_string(),
            _ => eprintln!("unknown arg: {arg}"),
        }
    }

    // Feature length: pixels per eye × 2 eyes (×2 if gradient appended) + 3 head pose
    let feat_per_eye = patch_w * patch_h * if use_gradient { 2 } else { 1 };
    let head_feat_len = if no_head_pose { 0 } else { 3 };
    let feat_len = feat_per_eye * 2 + head_feat_len;

    let feat_desc = if use_gradient { "CLAHE+SobelX" } else { "CLAHE" };
    let flip_desc = if flip_right { ", right-eye flipped" } else { "" };
    let calib_desc = if uniform_calib { "uniform" } else { "first" };
    let clahe_desc = format!("tiles={clahe_tx}×{clahe_ty} clip={clahe_clip}");
    println!("MPIIGaze benchmark — preprocessed root: {proc_root}");
    println!("Protocol: {calib_desc} {n_calib} frames/subject = calibration, rest = test");
    println!("Features: {feat_len}-D ({feat_desc} {patch_w}×{patch_h} [{clahe_desc}] eye patches + 3 head-pose{flip_desc})");
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

        if samples.len() < n_calib + 10 {
            eprintln!("  skip {subject}: only {} samples (need >{})", samples.len(), n_calib + 10);
            continue;
        }

        print!("  {subject}: extracting features ({} frames)...", samples.len());
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let extracted: Vec<Option<(Vec<f32>, f64, f64)>> = samples.iter()
            .map(|s| extract_features(s, patch_w, patch_h, flip_right, use_gradient, clahe_tx, clahe_ty, clahe_clip, no_head_pose))
            .collect();

        // Select calibration indices (first N, or uniformly spread across session)
        let n_total = extracted.len();
        let calib_indices: Vec<usize> = if uniform_calib {
            // Uniformly spread n_calib samples across the full session
            (0..n_calib)
                .map(|i| i * n_total / n_calib)
                .collect()
        } else {
            (0..n_calib).collect()
        };
        let calib_set: std::collections::HashSet<usize> = calib_indices.iter().copied().collect();

        let calib_count = calib_indices.iter()
            .filter(|&&i| extracted[i].is_some()).count();

        // Optional per-feature z-score normalization (computed on calibration set only)
        let (feat_mean, feat_std): (Vec<f32>, Vec<f32>) = if normalize_feats {
            let calib_feats: Vec<&Vec<f32>> = calib_indices.iter()
                .filter_map(|&i| extracted[i].as_ref().map(|(f, _, _)| f))
                .collect();
            let n = calib_feats.len();
            if n < 2 {
                (vec![0.0; feat_len], vec![1.0; feat_len])
            } else {
                let mut mean = vec![0.0f32; feat_len];
                for f in &calib_feats { for (j, &v) in f.iter().enumerate() { mean[j] += v; } }
                for m in &mut mean { *m /= n as f32; }
                let mut var = vec![0.0f32; feat_len];
                for f in &calib_feats { for (j, &v) in f.iter().enumerate() { var[j] += (v - mean[j]).powi(2); } }
                let std: Vec<f32> = var.iter().map(|&v| (v / n as f32).sqrt().max(1e-6)).collect();
                (mean, std)
            }
        } else {
            (vec![0.0; feat_len], vec![1.0; feat_len])
        };
        let normalize = |feats: &[f32]| -> Vec<f32> {
            if normalize_feats {
                feats.iter().enumerate().map(|(j, &v)| (v - feat_mean[j]) / feat_std[j]).collect()
            } else {
                feats.to_vec()
            }
        };

        let eye_feat_len = patch_w * patch_h;
        // Per-eye feature length for separate-eyes mode; head pose split between eyes
        let per_eye_feat = eye_feat_len + if no_head_pose { 0 } else { 1 }; // rough split

        let mut reg = RidgeRegressor::new(n_calib + 10, 1e4, feat_len);

        // For separate-eyes: two regressors, one per eye
        let mut reg_l = RidgeRegressor::new(n_calib + 10, 1e4, eye_feat_len);
        let mut reg_r = RidgeRegressor::new(n_calib + 10, 1e4, eye_feat_len);

        // Calibration phase
        for &i in &calib_indices {
            if let Some((feats, yaw, pitch)) = &extracted[i] {
                let feats_n = normalize(feats);
                let target_x = (*yaw * 1000.0) as f32;
                let target_y = (*pitch * 1000.0) as f32;
                if separate_eyes {
                    // Left eye: first eye_feat_len features
                    reg_l.add_sample(feats_n[..eye_feat_len].to_vec(), target_x, target_y);
                    // Right eye: next eye_feat_len features
                    reg_r.add_sample(feats_n[eye_feat_len..eye_feat_len*2].to_vec(), target_x, target_y);
                } else {
                    // Scale ×1000 so angles (≈ ±0.5 rad) are in a range where lambda auto-tuning works well
                    reg.add_sample(feats_n, target_x, target_y);
                }
            }
        }

        // Auto-tune lambda via LOO CV
        if separate_eyes {
            if let Some(lam) = reg_l.auto_lambda(LAMBDA_CANDIDATES) { reg_l.set_lambda(lam); }
            if let Some(lam) = reg_r.auto_lambda(LAMBDA_CANDIDATES) { reg_r.set_lambda(lam); }
        } else if let Some(best_lam) = reg.auto_lambda(LAMBDA_CANDIDATES) {
            reg.set_lambda(best_lam);
        }

        // Solve once — reuse coefficients for all test predictions (O(p) each vs O(n·p²))
        let (beta_x, beta_y, beta_lx, beta_ly, beta_rx, beta_ry);
        if separate_eyes {
            let (lx, ly) = reg_l.solve().unwrap_or_else(|| (vec![0.0; eye_feat_len], vec![0.0; eye_feat_len]));
            let (rx, ry) = reg_r.solve().unwrap_or_else(|| (vec![0.0; eye_feat_len], vec![0.0; eye_feat_len]));
            beta_x = vec![]; beta_y = vec![];
            beta_lx = lx; beta_ly = ly; beta_rx = rx; beta_ry = ry;
        } else {
            let Some((bx, by)) = reg.solve() else {
                println!(" solve failed"); continue;
            };
            beta_x = bx; beta_y = by;
            beta_lx = vec![]; beta_ly = vec![]; beta_rx = vec![]; beta_ry = vec![];
        }
        let _ = per_eye_feat;

        // Test phase: all frames NOT in calibration set
        let mut subject_errors: Vec<f64> = Vec::new();
        for (i, entry) in extracted.iter().enumerate() {
            if calib_set.contains(&i) { continue; }
            let Some((feats, true_yaw, true_pitch)) = entry else { continue };
            let feats_n = normalize(feats);
            let (py_scaled, pp_scaled) = if separate_eyes {
                let (lx, ly) = RidgeRegressor::predict_from_coeffs(&feats_n[..eye_feat_len], &beta_lx, &beta_ly);
                let (rx, ry) = RidgeRegressor::predict_from_coeffs(&feats_n[eye_feat_len..eye_feat_len*2], &beta_rx, &beta_ry);
                ((lx + rx) / 2.0, (ly + ry) / 2.0)
            } else {
                RidgeRegressor::predict_from_coeffs(&feats_n, &beta_x, &beta_y)
            };

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

/// Flip a grayscale image horizontally (mirror left↔right).
fn flip_horizontal(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = img[y * w + (w - 1 - x)];
        }
    }
    out
}

/// Compute per-pixel Sobel-x gradient on a CLAHE'd image (already [0,255]).
/// Returns values shifted to [0,255] (128 = zero gradient).
fn sobel_x_features(clahe: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![128.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xm = if x > 0 { x - 1 } else { 0 };
            let xp = if x + 1 < w { x + 1 } else { w - 1 };
            let ym = if y > 0 { y - 1 } else { 0 };
            let yp = if y + 1 < h { y + 1 } else { h - 1 };
            // Sobel-x: [-1 0 1; -2 0 2; -1 0 1]
            let gx = -1.0 * clahe[ym * w + xm]
                   +  1.0 * clahe[ym * w + xp]
                   + -2.0 * clahe[y  * w + xm]
                   +  2.0 * clahe[y  * w + xp]
                   + -1.0 * clahe[yp * w + xm]
                   +  1.0 * clahe[yp * w + xp];
            // Scale: max kernel response ≈ 8×255=2040; map to [0,255]
            out[y * w + x] = (gx / 2040.0 * 127.5 + 128.0).clamp(0.0, 255.0);
        }
    }
    out
}

/// Extract features from one sample. Returns (features, yaw_rad, pitch_rad) or None.
fn extract_features(
    s: &Sample,
    patch_w: usize,
    patch_h: usize,
    flip_right: bool,
    use_gradient: bool,
    clahe_tx: usize,
    clahe_ty: usize,
    clahe_clip: f32,
    no_head_pose: bool,
) -> Option<(Vec<f32>, f64, f64)> {
    let load_gray = |path: &str| -> Option<(Vec<u8>, usize, usize)> {
        let img = ImageReader::open(path).ok()?.decode().ok()?.into_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Some((img.into_raw(), w, h))
    };

    let (left_gray,  lw, lh) = load_gray(&s.left_path)?;
    let (right_gray_raw, rw, rh) = load_gray(&s.right_path)?;
    let right_gray = if flip_right {
        flip_horizontal(&right_gray_raw, rw, rh)
    } else {
        right_gray_raw
    };

    // Patches are single-channel (grayscale PNG from preprocessor)
    let l_feat = ridge::extract_eye_features_gray_sized_clahe(&left_gray,  lw, lh, patch_w, patch_h, clahe_tx, clahe_ty, clahe_clip);
    let r_feat = ridge::extract_eye_features_gray_sized_clahe(&right_gray, rw, rh, patch_w, patch_h, clahe_tx, clahe_ty, clahe_clip);

    // Optionally append Sobel-x gradient features (same spatial layout as pixels)
    let l_feat = if use_gradient {
        let grad = sobel_x_features(&l_feat, patch_w, patch_h);
        [l_feat, grad].concat()
    } else { l_feat };
    let r_feat = if use_gradient {
        let grad = sobel_x_features(&r_feat, patch_w, patch_h);
        [r_feat, grad].concat()
    } else { r_feat };

    // Convert 3-D unit gaze vector → yaw/pitch angles.
    // MPIIGaze convention: x=right, y=down, z=toward camera.
    // yaw   = atan2(-gx, -gz)  (positive = looking left from subject's POV)
    // pitch = asin(-gy)        (positive = looking up)
    let [gx, gy, gz] = s.gaze;
    let norm = (gx*gx + gy*gy + gz*gz).sqrt();
    if norm < 1e-6 { return None; }
    let yaw   = (-gx / norm).atan2(-gz / norm);
    let pitch = ((-gy) / norm).asin();

    let head_feat_len = if no_head_pose { 0 } else { 3 };
    let feat_len = patch_w * patch_h * 2 + head_feat_len;
    let mut feats = Vec::with_capacity(feat_len);
    feats.extend_from_slice(&l_feat);
    feats.extend_from_slice(&r_feat);
    if !no_head_pose {
        feats.push(s.head[0]);
        feats.push(s.head[1]);
        feats.push(s.head[2]);
    }

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
