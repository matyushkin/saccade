//! Offline benchmark: load a recorded session and try different parameters/algorithms.
//!
//! Usage: cargo run --release --example replay [path]
//! Default path: saccade_session.bin

use nalgebra::{DMatrix, DVector};
use saccade::ridge::RidgeRegressor;
use saccade::session::Session;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "saccade_session.bin".into());
    let session = match Session::load(&path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Failed to load {path}: {e}"); std::process::exit(1); }
    };

    println!("=== Loaded session: {path} ===");
    println!("Screen: {}x{}", session.screen_w, session.screen_h);
    println!("Calibration samples: {}", session.calibration.len());
    println!("Validation samples:  {}", session.validation.len());
    println!("Validation target:   ({:.0}, {:.0})", session.validation_target.0, session.validation_target.1);
    println!();

    if session.calibration.is_empty() || session.validation.is_empty() {
        eprintln!("Empty session — nothing to benchmark");
        std::process::exit(1);
    }

    let feat_len = session.calibration[0].features.len();
    println!("Feature length: {feat_len}");
    println!();

    // === Experiment 1: ridge lambda sweep ===
    println!("=== Ridge lambda sweep (no smoothing) ===");
    println!("{:>14} | {:>10} | {:>10} | {:>10}", "lambda", "mean_err", "median", "std (jitter)");
    println!("{}", "-".repeat(56));
    let lambdas = [1e3, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7, 1e8];
    for &lam in &lambdas {
        let result = bench_ridge(&session, lam, feat_len);
        println!(
            "{:>14.2e} | {:>10.0} | {:>10.0} | {:>10.0}",
            lam, result.mean_err, result.median_err, result.std_jitter
        );
    }
    println!();

    // === Experiment 2: buffer size (use all vs subset) ===
    println!("=== Calibration buffer subset ===");
    println!("{:>14} | {:>10} | {:>10}", "first_n", "mean_err", "std_jitter");
    println!("{}", "-".repeat(40));
    for &take in &[5usize, 10, 20, 30, 45, 90, 200, session.calibration.len()] {
        if take > session.calibration.len() { continue; }
        let result = bench_ridge_subset(&session, 1e-5, feat_len, take);
        println!("{:>14} | {:>10.0} | {:>10.0}", take, result.mean_err, result.std_jitter);
    }
    println!();

    // === Experiment 3: temporal smoothing ===
    println!("=== Smoothing (ridge λ=1e-5) ===");
    println!("{:>20} | {:>10} | {:>10}", "method", "mean_err", "std_jitter");
    println!("{}", "-".repeat(46));
    bench_smoothed(&session, 1e-5, feat_len, "raw", 0);
    bench_smoothed(&session, 1e-5, feat_len, "moving_avg(4)", 4);
    bench_smoothed(&session, 1e-5, feat_len, "moving_avg(8)", 8);
    bench_smoothed(&session, 1e-5, feat_len, "moving_avg(16)", 16);
    bench_smoothed(&session, 1e-5, feat_len, "moving_avg(32)", 32);
    println!();

    // === Experiment 4: feature normalization ===
    println!("=== Feature normalization ===");
    println!("{:>20} | {:>10} | {:>10}", "scheme", "mean_err", "std_jitter");
    println!("{}", "-".repeat(46));
    bench_normalization(&session, "raw", false, false);
    bench_normalization(&session, "div_255", true, false);
    bench_normalization(&session, "zero_mean", false, true);
    bench_normalization(&session, "div_255+mean", true, true);
    println!();

    // === Experiment 5: bias term (add a 1.0 to features) ===
    println!("=== Bias term (intercept) ===");
    println!("{:>20} | {:>10} | {:>10}", "scheme", "mean_err", "std_jitter");
    println!("{}", "-".repeat(46));
    bench_with_bias(&session, "no_bias", false);
    bench_with_bias(&session, "with_bias", true);
    println!();

    // === Experiment 6: outlier rejection on calibration ===
    println!("=== Calibration outlier rejection (drop residuals > k·σ) ===");
    println!("{:>20} | {:>10} | {:>10}", "k_sigma", "mean_err", "std_jitter");
    println!("{}", "-".repeat(46));
    for &k in &[10.0f64, 3.0, 2.0, 1.5, 1.0] {
        bench_outlier_rejection(&session, k);
    }
    println!();

    // === Experiment 7: cross-validation accuracy ===
    println!("=== Leave-one-out cross-validation on calibration set ===");
    println!("Mean LOO error: {:.0} px (raw 120-D features)", bench_loo_cv(&session, 1e-5));
    println!();

    // === Experiment 8: PCA dimensionality reduction ===
    println!("=== PCA reduction (test against overfitting) ===");
    println!("{:>10} | {:>10} | {:>10} | {:>10}", "n_components", "valid_err", "loo_err", "std_jit");
    println!("{}", "-".repeat(48));
    for &k in &[2usize, 4, 6, 8, 10, 12, 16, 24, 36] {
        if k > session.calibration.len() { continue; }
        let r = bench_pca(&session, k);
        println!("{:>10} | {:>10.0} | {:>10.0} | {:>10.0}", k, r.0, r.1, r.2);
    }
}

/// PCA reduce features to k dimensions, train ridge, return (valid_err, loo_err, std_jitter)
fn bench_pca(session: &Session, k: usize) -> (f64, f64, f64) {
    let n = session.calibration.len();
    let p = session.calibration[0].features.len();

    // Build calibration matrix (n × p)
    let mut x_data = Vec::with_capacity(n * p);
    for c in &session.calibration {
        for &f in &c.features {
            x_data.push(f as f64);
        }
    }
    let x_mat = DMatrix::from_row_slice(n, p, &x_data);

    // Center features
    let mean_row: DVector<f64> = DVector::from_iterator(p,
        (0..p).map(|j| x_mat.column(j).mean()));
    let x_centered = x_mat.clone() - DMatrix::from_fn(n, p, |_, j| mean_row[j]);

    // SVD: x_centered = U Σ Vᵀ
    let svd = x_centered.clone().svd(true, true);
    let v_t = svd.v_t.unwrap();

    // Top k right singular vectors (columns of V) — these are PCA loadings
    let k_use = k.min(v_t.nrows());
    let pca_basis = v_t.rows(0, k_use).transpose(); // p × k

    // Project calibration features to k-D
    let calib_proj = &x_centered * &pca_basis; // n × k

    // Train ridge in k-D space
    let mut sess2 = session.clone();
    sess2.calibration.clear();
    for i in 0..n {
        let row = calib_proj.row(i);
        let feats: Vec<f32> = row.iter().map(|&v| v as f32).collect();
        sess2.calibration.push(saccade::session::CalibFrame {
            features: feats,
            target_x: session.calibration[i].target_x,
            target_y: session.calibration[i].target_y,
        });
    }

    // Project validation features the same way
    sess2.validation.clear();
    for v in &session.validation {
        let centered: DVector<f64> = DVector::from_iterator(p,
            v.features.iter().enumerate().map(|(j, &f)| f as f64 - mean_row[j]));
        let proj = pca_basis.transpose() * centered;
        let feats: Vec<f32> = proj.iter().map(|&v| v as f32).collect();
        sess2.validation.push(saccade::session::ValidFrame { features: feats });
    }

    let r = bench_ridge(&sess2, 1e-5, k_use);
    let loo = bench_loo_cv(&sess2, 1e-5);
    (r.mean_err, loo, r.std_jitter)
}

fn bench_with_bias(session: &Session, label: &str, add_bias: bool) {
    let mut sess2 = session.clone();
    if add_bias {
        for c in sess2.calibration.iter_mut() { c.features.push(1.0); }
        for v in sess2.validation.iter_mut() { v.features.push(1.0); }
    }
    let feat_len = sess2.calibration[0].features.len();
    let r = bench_ridge(&sess2, 1e-5, feat_len);
    println!("{:>20} | {:>10.0} | {:>10.0}", label, r.mean_err, r.std_jitter);
}

fn bench_outlier_rejection(session: &Session, k_sigma: f64) {
    use saccade::ridge::RidgeRegressor;
    let feat_len = session.calibration[0].features.len();
    let mut reg = RidgeRegressor::new(session.calibration.len(), 1e-5, feat_len);
    for c in &session.calibration {
        reg.add_sample(c.features.clone(), c.target_x, c.target_y);
    }

    // Compute residuals on calibration set itself
    let mut residuals: Vec<f64> = Vec::new();
    for c in &session.calibration {
        if let Some((px, py)) = reg.predict(&c.features) {
            let dx = px as f64 - c.target_x as f64;
            let dy = py as f64 - c.target_y as f64;
            residuals.push((dx*dx + dy*dy).sqrt());
        }
    }
    if residuals.is_empty() {
        println!("{:>20} | {:>10} | {:>10}", format!("{k_sigma}σ"), "—", "—");
        return;
    }
    let n = residuals.len() as f64;
    let mean_r = residuals.iter().sum::<f64>() / n;
    let std_r = (residuals.iter().map(|r| (r-mean_r).powi(2)).sum::<f64>() / n).sqrt();
    let threshold = mean_r + k_sigma * std_r;

    // Refit without outliers
    let mut reg2 = RidgeRegressor::new(session.calibration.len(), 1e-5, feat_len);
    let mut kept = 0;
    for (c, r) in session.calibration.iter().zip(residuals.iter()) {
        if *r < threshold {
            reg2.add_sample(c.features.clone(), c.target_x, c.target_y);
            kept += 1;
        }
    }

    let target = session.validation_target;
    let mut errors = Vec::new();
    let mut preds = Vec::new();
    for v in &session.validation {
        if let Some((px, py)) = reg2.predict(&v.features) {
            let dx = px as f64 - target.0 as f64;
            let dy = py as f64 - target.1 as f64;
            errors.push((dx*dx + dy*dy).sqrt());
            preds.push((px as f64, py as f64));
        }
    }
    if errors.is_empty() { return; }
    let mn = errors.iter().sum::<f64>() / errors.len() as f64;
    let n2 = preds.len() as f64;
    let mx = preds.iter().map(|p| p.0).sum::<f64>() / n2;
    let my = preds.iter().map(|p| p.1).sum::<f64>() / n2;
    let std_j = (preds.iter().map(|p| (p.0-mx).powi(2)+(p.1-my).powi(2)).sum::<f64>() / n2).sqrt();
    let label = if k_sigma.is_infinite() { format!("none ({} kept)", kept) }
                else { format!("{}σ ({} kept)", k_sigma, kept) };
    println!("{:>20} | {:>10.0} | {:>10.0}", label, mn, std_j);
}

fn bench_loo_cv(session: &Session, _lambda: f64) -> f64 {
    use saccade::ridge::RidgeRegressor;
    let feat_len = session.calibration[0].features.len();
    let mut errors = Vec::new();
    for i in 0..session.calibration.len() {
        let mut reg = RidgeRegressor::new(session.calibration.len(), 1e-5, feat_len);
        for (j, c) in session.calibration.iter().enumerate() {
            if j != i {
                reg.add_sample(c.features.clone(), c.target_x, c.target_y);
            }
        }
        let held = &session.calibration[i];
        if let Some((px, py)) = reg.predict(&held.features) {
            let dx = px as f64 - held.target_x as f64;
            let dy = py as f64 - held.target_y as f64;
            errors.push((dx*dx + dy*dy).sqrt());
        }
    }
    if errors.is_empty() { return f64::NAN; }
    errors.iter().sum::<f64>() / errors.len() as f64
}

struct BenchResult {
    mean_err: f64,
    median_err: f64,
    std_jitter: f64,
}

fn bench_ridge(session: &Session, lambda: f64, feat_len: usize) -> BenchResult {
    bench_ridge_subset(session, lambda, feat_len, session.calibration.len())
}

fn bench_ridge_subset(session: &Session, lambda: f64, feat_len: usize, take: usize) -> BenchResult {
    let mut reg = RidgeRegressor::new(take.max(1), lambda, feat_len);
    for c in session.calibration.iter().take(take) {
        reg.add_sample(c.features.clone(), c.target_x, c.target_y);
    }

    let target = session.validation_target;
    let mut errors: Vec<f64> = Vec::new();
    let mut predictions: Vec<(f64, f64)> = Vec::new();
    for v in &session.validation {
        if let Some((px, py)) = reg.predict(&v.features) {
            let dx = px as f64 - target.0 as f64;
            let dy = py as f64 - target.1 as f64;
            errors.push((dx * dx + dy * dy).sqrt());
            predictions.push((px as f64, py as f64));
        }
    }

    if errors.is_empty() {
        return BenchResult { mean_err: f64::NAN, median_err: f64::NAN, std_jitter: f64::NAN };
    }
    let n = errors.len() as f64;
    let mean_err = errors.iter().sum::<f64>() / n;
    let mut sorted = errors.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_err = sorted[sorted.len() / 2];

    // Jitter = std of predictions around their own mean
    let mx = predictions.iter().map(|p| p.0).sum::<f64>() / n;
    let my = predictions.iter().map(|p| p.1).sum::<f64>() / n;
    let var = predictions.iter().map(|p| (p.0 - mx).powi(2) + (p.1 - my).powi(2)).sum::<f64>() / n;
    let std_jitter = var.sqrt();

    BenchResult { mean_err, median_err, std_jitter }
}

fn bench_smoothed(session: &Session, lambda: f64, feat_len: usize, label: &str, window: usize) {
    let mut reg = RidgeRegressor::new(session.calibration.len(), lambda, feat_len);
    for c in &session.calibration {
        reg.add_sample(c.features.clone(), c.target_x, c.target_y);
    }

    let target = session.validation_target;
    let mut buffer: Vec<(f64, f64)> = Vec::new();
    let mut errors = Vec::new();
    let mut smoothed_predictions: Vec<(f64, f64)> = Vec::new();

    for v in &session.validation {
        if let Some((px, py)) = reg.predict(&v.features) {
            let raw = (px as f64, py as f64);
            let smoothed = if window == 0 {
                raw
            } else {
                buffer.push(raw);
                if buffer.len() > window { buffer.remove(0); }
                let n = buffer.len() as f64;
                let mx = buffer.iter().map(|p| p.0).sum::<f64>() / n;
                let my = buffer.iter().map(|p| p.1).sum::<f64>() / n;
                (mx, my)
            };
            let dx = smoothed.0 - target.0 as f64;
            let dy = smoothed.1 - target.1 as f64;
            errors.push((dx * dx + dy * dy).sqrt());
            smoothed_predictions.push(smoothed);
        }
    }

    if errors.is_empty() {
        println!("{:>20} | {:>10} | {:>10}", label, "—", "—");
        return;
    }
    let n = errors.len() as f64;
    let mean_err = errors.iter().sum::<f64>() / n;
    let mx = smoothed_predictions.iter().map(|p| p.0).sum::<f64>() / n;
    let my = smoothed_predictions.iter().map(|p| p.1).sum::<f64>() / n;
    let std_jitter = (smoothed_predictions.iter()
        .map(|p| (p.0 - mx).powi(2) + (p.1 - my).powi(2)).sum::<f64>() / n).sqrt();

    println!("{:>20} | {:>10.0} | {:>10.0}", label, mean_err, std_jitter);
}

fn bench_normalization(session: &Session, label: &str, divide_255: bool, zero_mean: bool) {
    // Create a normalized copy
    let mut sess2 = session.clone();
    let normalize = |v: &mut Vec<f32>| {
        if divide_255 { for x in v.iter_mut() { *x /= 255.0; } }
        if zero_mean {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            for x in v.iter_mut() { *x -= mean; }
        }
    };
    for c in sess2.calibration.iter_mut() { normalize(&mut c.features); }
    for v in sess2.validation.iter_mut() { normalize(&mut v.features); }

    let feat_len = sess2.calibration[0].features.len();
    let r = bench_ridge(&sess2, 1e-5, feat_len);
    println!("{:>20} | {:>10.0} | {:>10.0}", label, r.mean_err, r.std_jitter);
}
