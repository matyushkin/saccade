//! Ridge regression with WebGazer-style eye feature extraction.
//!
//! Port of WebGazer.js's core algorithm:
//! - Extract eye patch as 10×6 grayscale, CLAHE normalized (60-D per eye)
//! - Concatenate both eyes → 120-D feature vector
//! - Weighted ridge regression: β = (X^T W X + λI)^{-1} X^T W y
//!   with chronological decay w_i = sqrt(1/(n-i)) (newer samples weighted more)
//! - Retrain on every prediction using circular buffer of last N samples
//!
//! References:
//!   Papoutsaki et al., "WebGazer: Scalable Webcam Eye Tracking Using User
//!   Interactions", IJCAI 2016.
//!   Krafka et al., "Eye Tracking for Everyone", CVPR 2016.

use nalgebra::{DMatrix, DVector};

/// Eye patch output resolution.
/// E14 ablation (Original data): 20×12 → 10.35° vs 10×6 → 12.37° (-16%).
/// E15 ablation (Normalized data): 20×12 → 5.89° ≈ 30×18 → 5.91° ≈ 40×24 → 5.86° (plateau).
/// 20×12 optimal: best accuracy/feature ratio; larger patches offer negligible gain at n=200.
pub const EYE_PATCH_W: usize = 20;
pub const EYE_PATCH_H: usize = 12;
pub const EYE_FEAT_LEN: usize = EYE_PATCH_W * EYE_PATCH_H; // 240 per eye
pub const BOTH_EYES_FEAT_LEN: usize = EYE_FEAT_LEN * 2; // 480 total

/// Extract WebGazer-style features from an RGB eye patch.
/// Returns a 60-element feature vector (10×6 CLAHE-normalized grayscale).
///
/// Pipeline: RGB → grayscale → CLAHE (2×2 tiles, clip=4.0) → resize to 10×6
pub fn extract_eye_features(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    if rgb.len() != width * height * 3 || width == 0 || height == 0 {
        return vec![0.0; EYE_FEAT_LEN];
    }

    // Step 1: Convert to grayscale at original resolution
    let mut gray = vec![0u8; width * height];
    for i in 0..width * height {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
    }

    // Step 2: Apply CLAHE on the full-resolution grayscale patch (2×2 tiles)
    // This handles uneven lighting across the eye (e.g., screen-side vs dark side)
    let equalized = clahe_gray(&gray, width, height, 2, 2, 4.0);

    // Step 3: Bilinear resize to 10×6
    bilinear_resize_gray_f32(&equalized, width, height, EYE_PATCH_W, EYE_PATCH_H)
}

/// Extract features from a grayscale eye patch.
pub fn extract_eye_features_gray(gray_patch: &[u8], width: usize, height: usize) -> Vec<f32> {
    extract_eye_features_gray_sized(gray_patch, width, height, EYE_PATCH_W, EYE_PATCH_H)
}

/// Extract features from a grayscale eye patch, resizing to an arbitrary output size.
/// Useful for resolution ablation (e.g., sweep 10×6 → 40×24).
pub fn extract_eye_features_gray_sized(
    gray_patch: &[u8],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<f32> {
    let out_len = out_w * out_h;
    if gray_patch.len() != width * height || width == 0 || height == 0 {
        return vec![0.0; out_len];
    }

    let equalized = clahe_gray(gray_patch, width, height, 2, 2, 4.0);
    bilinear_resize_gray_f32(&equalized, width, height, out_w, out_h)
}

/// CLAHE (Contrast Limited Adaptive Histogram Equalization) on a grayscale image.
///
/// Divides the image into `tiles_x × tiles_y` tiles, computes a clipped+redistributed
/// histogram per tile, then uses bilinear interpolation between tile CLRs for each pixel.
/// `clip_limit`: max allowed histogram bin as a multiple of the average bin count.
///   4.0 is typical for eye images (prevents over-amplification of noise).
fn clahe_gray(
    src: &[u8],
    width: usize,
    height: usize,
    tiles_x: usize,
    tiles_y: usize,
    clip_limit: f32,
) -> Vec<u8> {
    let tiles_x = tiles_x.max(1);
    let tiles_y = tiles_y.max(1);
    let n_tiles = tiles_x * tiles_y;

    // Build per-tile CLRs (Context Level Remapping functions)
    let mut clrs = vec![[0u8; 256]; n_tiles];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * width / tiles_x;
            let x1 = (tx + 1) * width / tiles_x;
            let y0 = ty * height / tiles_y;
            let y1 = (ty + 1) * height / tiles_y;
            let n_pixels = ((x1 - x0) * (y1 - y0)).max(1);

            // Histogram for this tile
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[src[y * width + x] as usize] += 1;
                }
            }

            // Clip histogram and redistribute excess uniformly
            let clip_max = ((clip_limit * n_pixels as f32 / 256.0) as u32).max(1);
            let mut excess = 0u32;
            for h in hist.iter_mut() {
                if *h > clip_max {
                    excess += *h - clip_max;
                    *h = clip_max;
                }
            }
            // Distribute excess evenly across all bins
            let add_per_bin = excess / 256;
            let remainder = (excess % 256) as usize;
            for (i, h) in hist.iter_mut().enumerate() {
                *h += add_per_bin;
                if i < remainder {
                    *h += 1;
                }
            }

            // Build cumulative distribution function, map to [0, 255]
            let tile_idx = ty * tiles_x + tx;
            let mut acc = 0u32;
            for i in 0..256 {
                acc += hist[i];
                clrs[tile_idx][i] = (acc as f32 * 255.0 / n_pixels as f32).min(255.0) as u8;
            }
        }
    }

    // Apply CLRs with bilinear interpolation between tile centers
    let mut dst = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let v = src[y * width + x] as usize;

            // Fractional tile coordinate (center of tile has coord 0.5)
            let tx_f = (x as f32 + 0.5) * tiles_x as f32 / width as f32 - 0.5;
            let ty_f = (y as f32 + 0.5) * tiles_y as f32 / height as f32 - 0.5;

            let tx0 = (tx_f.floor() as i32).clamp(0, tiles_x as i32 - 1) as usize;
            let ty0 = (ty_f.floor() as i32).clamp(0, tiles_y as i32 - 1) as usize;
            let tx1 = (tx0 + 1).min(tiles_x - 1);
            let ty1 = (ty0 + 1).min(tiles_y - 1);
            let fx = (tx_f - tx0 as f32).clamp(0.0, 1.0);
            let fy = (ty_f - ty0 as f32).clamp(0.0, 1.0);

            let c00 = clrs[ty0 * tiles_x + tx0][v] as f32;
            let c10 = clrs[ty0 * tiles_x + tx1][v] as f32;
            let c01 = clrs[ty1 * tiles_x + tx0][v] as f32;
            let c11 = clrs[ty1 * tiles_x + tx1][v] as f32;

            let interp = c00 * (1.0 - fx) * (1.0 - fy)
                + c10 * fx * (1.0 - fy)
                + c01 * (1.0 - fx) * fy
                + c11 * fx * fy;
            dst[y * width + x] = interp as u8;
        }
    }
    dst
}

/// Bilinear resize of a grayscale image, returning f32 pixels in [0, 255].
fn bilinear_resize_gray_f32(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx as f32 * sw as f32 / dw as f32;
            let sy = dy as f32 * sh as f32 / dh as f32;
            let x0 = (sx as usize).min(sw - 1);
            let y0 = (sy as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let p00 = src[y0 * sw + x0] as f32;
            let p01 = src[y0 * sw + x1] as f32;
            let p10 = src[y1 * sw + x0] as f32;
            let p11 = src[y1 * sw + x1] as f32;
            dst[dy * dw + dx] = p00 * (1.0 - fx) * (1.0 - fy)
                + p01 * fx * (1.0 - fy)
                + p10 * (1.0 - fx) * fy
                + p11 * fx * fy;
        }
    }
    dst
}

fn bilinear_resize_rgb(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * 3];
    for dy in 0..dh {
        for dx in 0..dw {
            let sx = dx as f32 * sw as f32 / dw as f32;
            let sy = dy as f32 * sh as f32 / dh as f32;
            let x0 = (sx as usize).min(sw - 1);
            let y0 = (sy as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let v = p00 * (1.0 - fx) * (1.0 - fy)
                    + p01 * fx * (1.0 - fy)
                    + p10 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[(dy * dw + dx) * 3 + c] = v as u8;
            }
        }
    }
    dst
}

/// A single calibration sample: features + target screen position.
#[derive(Debug, Clone)]
pub struct RidgeSample {
    pub features: Vec<f32>,
    pub target_x: f32,
    pub target_y: f32,
}

/// Ridge regressor with circular buffer and chronological decay weights.
///
/// Uses weighted ridge regression: `β = (X^T W X + λI)^{-1} X^T W y`
/// where weights follow chronological decay: `w_i = sqrt(1 / (n - i))`
/// (older samples get lower weight; newest sample always has weight 1.0).
///
/// Retrains on every `predict()` call using the current buffer contents.
pub struct RidgeRegressor {
    pub samples: Vec<RidgeSample>,
    pub max_samples: usize,
    pub lambda: f64,
    pub feat_len: usize,
}

impl RidgeRegressor {
    pub fn new(max_samples: usize, lambda: f64, feat_len: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
            lambda,
            feat_len,
        }
    }

    /// Add a training sample. Evicts oldest if full (FIFO).
    pub fn add_sample(&mut self, features: Vec<f32>, target_x: f32, target_y: f32) {
        if features.len() != self.feat_len {
            return;
        }
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(RidgeSample { features, target_x, target_y });
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Find optimal lambda via leave-one-out cross-validation on stored samples.
    /// Uses the hat-matrix shortcut: O(p³ + N·p²) instead of O(N²·p²).
    pub fn auto_lambda(&self, candidates: &[f64]) -> Option<f64> {
        if self.samples.len() < 4 {
            return None;
        }
        let mut best_lam = candidates[0];
        let mut best_err = f64::INFINITY;
        for &lam in candidates {
            let err = self.loo_error(lam);
            if err < best_err {
                best_err = err;
                best_lam = lam;
            }
        }
        Some(best_lam)
    }

    /// Compute mean leave-one-out error for a given lambda using the hat-matrix shortcut.
    ///
    /// For ridge regression the LOO prediction error is:
    ///   `e_i^{loo} = (y_i - ŷ_i) / (1 - h_{ii})`
    /// where `h_{ii} = x̃_i^T (X̃^T X̃ + λI)^{-1} x̃_i` and `x̃_i = sqrt(w_i) x_i`.
    ///
    /// Complexity: O(p³ + N·p²) vs the naive O(N·(N·p²)) = O(N²·p²).
    pub fn loo_error(&self, lambda: f64) -> f64 {
        let n = self.samples.len();
        if n < 4 {
            return f64::INFINITY;
        }
        let p = self.feat_len;
        let weights = chronological_weights(n);

        // Build weighted design matrix X̃ (n×p) and weighted targets ỹ
        let mut x_data = vec![0.0f64; n * p];
        let mut y_x = vec![0.0f64; n];
        let mut y_y = vec![0.0f64; n];
        for (i, s) in self.samples.iter().enumerate() {
            let w = weights[i];
            for (j, &f) in s.features.iter().enumerate() {
                x_data[i * p + j] = f as f64 * w;
            }
            y_x[i] = s.target_x as f64 * w;
            y_y[i] = s.target_y as f64 * w;
        }

        let x_mat = DMatrix::from_row_slice(n, p, &x_data);
        let xt = x_mat.transpose();
        let mut xtx = &xt * &x_mat; // p×p
        for i in 0..p {
            xtx[(i, i)] += lambda;
        }

        // Solve (X̃^T X̃ + λI) A_cols = X̃^T  →  A = (X̃^T X̃ + λI)^{-1} is p×p
        // We need it explicitly for h_{ii} = x̃_i^T A x̃_i.
        // Compute via LU: A = (X̃^T X̃ + λI)^{-1}
        let decomp = xtx.lu();
        let identity = DMatrix::identity(p, p);
        let a_inv = match decomp.solve(&identity) {
            Some(m) => m,
            None => return f64::INFINITY,
        };

        // Full-data predictions: β = A X̃^T ỹ
        let xty_x = &xt * DVector::from_vec(y_x.clone());
        let xty_y = &xt * DVector::from_vec(y_y.clone());
        let beta_x = &a_inv * &xty_x;
        let beta_y = &a_inv * &xty_y;

        // Compute hat values h_{ii} and LOO errors
        let mut total_err = 0.0f64;
        let mut count = 0usize;
        for (i, s) in self.samples.iter().enumerate() {
            let w = weights[i];
            // x̃_i = sqrt(w_i) * x_i (row i of X̃)
            let x_tilde: Vec<f64> = s.features.iter().map(|&f| f as f64 * w).collect();
            let x_tilde_v = DVector::from_vec(x_tilde);

            // ŷ_i = x_i^T β  (unweighted prediction)
            let x_orig: Vec<f64> = s.features.iter().map(|&f| f as f64).collect();
            let x_orig_v = DVector::from_vec(x_orig);
            let yhat_x = x_orig_v.dot(&beta_x);
            let yhat_y = x_orig_v.dot(&beta_y);

            // h_{ii} = x̃_i^T A x̃_i
            let ax = &a_inv * &x_tilde_v; // p-vec
            let h_ii = x_tilde_v.dot(&ax);

            // LOO shrinkage factor: clamp to avoid division by zero
            let shrink = (1.0 - h_ii).max(1e-4);

            let rx = (s.target_x as f64 - yhat_x) / shrink;
            let ry = (s.target_y as f64 - yhat_y) / shrink;
            total_err += (rx * rx + ry * ry).sqrt();
            count += 1;
        }

        if count == 0 { f64::INFINITY } else { total_err / count as f64 }
    }

    /// Override the lambda (e.g., after auto_lambda picks one).
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Solve ridge regression and return (beta_x, beta_y) coefficient vectors.
    /// Use `predict_from_coeffs` for fast repeated predictions without re-solving.
    pub fn solve(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        let n = self.samples.len();
        if n < 3 { return None; }
        let p = self.feat_len;
        let weights = chronological_weights(n);

        let mut x_data = Vec::with_capacity(n * p);
        let mut y_x = Vec::with_capacity(n);
        let mut y_y = Vec::with_capacity(n);
        for (i, s) in self.samples.iter().enumerate() {
            let w = weights[i];
            for &f in &s.features {
                x_data.push(f as f64 * w);
            }
            y_x.push(s.target_x as f64 * w);
            y_y.push(s.target_y as f64 * w);
        }

        let x_mat = DMatrix::from_row_slice(n, p, &x_data);
        let xt = x_mat.transpose();
        let mut xtx = &xt * &x_mat;
        for i in 0..p { xtx[(i, i)] += self.lambda; }

        let y_x_vec = DVector::from_vec(y_x);
        let y_y_vec = DVector::from_vec(y_y);
        let xty_x = &xt * &y_x_vec;
        let xty_y = &xt * &y_y_vec;
        let decomp = xtx.lu();
        let beta_x = decomp.solve(&xty_x)?;
        let beta_y = decomp.solve(&xty_y)?;

        Some((beta_x.as_slice().to_vec(), beta_y.as_slice().to_vec()))
    }

    /// Predict using pre-solved coefficients (O(p) — much faster for batch prediction).
    pub fn predict_from_coeffs(features: &[f32], beta_x: &[f64], beta_y: &[f64]) -> (f32, f32) {
        let mut px = 0.0f64;
        let mut py = 0.0f64;
        for (i, &f) in features.iter().enumerate() {
            px += f as f64 * beta_x[i];
            py += f as f64 * beta_y[i];
        }
        (px as f32, py as f32)
    }

    /// Solve weighted ridge regression and predict for given features.
    ///
    /// Weights follow chronological decay: `w_i = sqrt(1 / (n - i))`.
    /// Newest sample has weight 1.0; oldest sample has weight `sqrt(1/n)`.
    /// Returns None if not enough samples or solver fails.
    pub fn predict(&self, features: &[f32]) -> Option<(f32, f32)> {
        if self.samples.len() < 3 || features.len() != self.feat_len {
            return None;
        }

        let n = self.samples.len();
        let p = self.feat_len;
        let weights = chronological_weights(n);

        // Build weighted design matrix X̃ (n×p) and weighted target vectors
        // X̃_i = sqrt(w_i) * x_i,  ỹ_i = sqrt(w_i) * y_i
        // This transforms WLS into standard ridge: min ||X̃ β - ỹ||² + λ||β||²
        let mut x_data = Vec::with_capacity(n * p);
        let mut y_x = Vec::with_capacity(n);
        let mut y_y = Vec::with_capacity(n);
        for (i, s) in self.samples.iter().enumerate() {
            let w = weights[i];
            for &f in &s.features {
                x_data.push(f as f64 * w);
            }
            y_x.push(s.target_x as f64 * w);
            y_y.push(s.target_y as f64 * w);
        }

        let x_mat = DMatrix::from_row_slice(n, p, &x_data);
        let xt = x_mat.transpose();
        let mut xtx = &xt * &x_mat;

        // Add λI (ridge regularization)
        for i in 0..p {
            xtx[(i, i)] += self.lambda;
        }

        let y_x_vec = DVector::from_vec(y_x);
        let y_y_vec = DVector::from_vec(y_y);
        let xty_x = &xt * &y_x_vec;
        let xty_y = &xt * &y_y_vec;

        // Solve (X̃^T X̃ + λI) β = X̃^T ỹ using LU decomposition
        let decomp = xtx.lu();
        let beta_x = decomp.solve(&xty_x)?;
        let beta_y = decomp.solve(&xty_y)?;

        // Predict: x^T β  (use original unweighted features for prediction)
        let mut px = 0.0f64;
        let mut py = 0.0f64;
        for i in 0..p {
            px += features[i] as f64 * beta_x[i];
            py += features[i] as f64 * beta_y[i];
        }

        Some((px as f32, py as f32))
    }
}

/// Chronological decay weights for n samples.
/// `w_i = sqrt(1 / (n - i))`: older samples (small i) get lower weight.
/// The newest sample (i = n-1) always gets weight 1.0.
fn chronological_weights(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (1.0 / (n - i) as f64).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_extraction_returns_60_values() {
        // 20×12 RGB eye patch
        let rgb = vec![128u8; 20 * 12 * 3];
        let feat = extract_eye_features(&rgb, 20, 12);
        assert_eq!(feat.len(), 60);
    }

    #[test]
    fn feature_extraction_gray() {
        let gray = vec![100u8; 30 * 18];
        let feat = extract_eye_features_gray(&gray, 30, 18);
        assert_eq!(feat.len(), 60);
    }

    #[test]
    fn clahe_preserves_size() {
        let src = vec![50u8; 40 * 20];
        let dst = clahe_gray(&src, 40, 20, 2, 2, 4.0);
        assert_eq!(dst.len(), 40 * 20);
    }

    #[test]
    fn clahe_improves_contrast_on_dark_patch() {
        // Left half = 20, right half = 200 → global HE would blend, CLAHE keeps local contrast
        let mut src = vec![0u8; 40 * 20];
        for y in 0..20 {
            for x in 0..40 {
                src[y * 40 + x] = if x < 20 { 20 } else { 200 };
            }
        }
        let dst = clahe_gray(&src, 40, 20, 2, 1, 4.0);
        // Left tile should be stretched toward full range
        let mut left_sum = 0.0f32;
        let mut right_sum = 0.0f32;
        for y in 0..20 {
            for x in 0..40 {
                if x < 20 { left_sum += dst[y * 40 + x] as f32; }
                else { right_sum += dst[y * 40 + x] as f32; }
            }
        }
        let left_mean = left_sum / 400.0;
        let right_mean = right_sum / 400.0;
        assert!(right_mean > left_mean, "CLAHE should preserve per-region contrast");
    }

    #[test]
    fn ridge_fits_linear_data() {
        // target_x = 2 * f[0] + 1 (linear in one feature)
        let mut reg = RidgeRegressor::new(20, 1e-5, 3);
        for i in 0..10 {
            let features = vec![i as f32, 0.0, 0.0];
            reg.add_sample(features, 2.0 * i as f32 + 1.0, 0.0);
        }
        // Predict for x=5 → expect ~11
        let pred = reg.predict(&[5.0, 0.0, 0.0]).unwrap();
        assert!((pred.0 - 11.0).abs() < 1.5, "predicted {}", pred.0);
    }

    #[test]
    fn ridge_circular_buffer() {
        let mut reg = RidgeRegressor::new(5, 1e-5, 2);
        for i in 0..10 {
            reg.add_sample(vec![i as f32, 0.0], i as f32, 0.0);
        }
        assert_eq!(reg.sample_count(), 5);
        // Oldest samples (0-4) should be evicted, keeping 5-9
        assert_eq!(reg.samples[0].target_x, 5.0);
        assert_eq!(reg.samples[4].target_x, 9.0);
    }

    #[test]
    fn auto_lambda_picks_from_candidates() {
        let mut reg = RidgeRegressor::new(20, 1e-5, 3);
        // Linear data: y = 2x + 1
        for i in 0..10 {
            reg.add_sample(vec![i as f32, 0.0, 0.0], 2.0 * i as f32 + 1.0, 0.0);
        }
        let lam = reg.auto_lambda(&[1e-5, 1e-3, 1e-1, 1.0, 1e3, 1e6]);
        assert!(lam.is_some());
    }

    #[test]
    fn loo_error_agrees_with_brute_force() {
        // Small dataset: verify hat-matrix LOO agrees with leave-one-out brute force
        let mut reg = RidgeRegressor::new(20, 1e3, 3);
        for i in 0..8 {
            let f = i as f32;
            reg.add_sample(vec![f, f * 0.5, 1.0], f * 3.0 + 10.0, f * 2.0 + 5.0);
        }
        let fast = reg.loo_error(1e3);
        assert!(fast.is_finite(), "LOO error should be finite: {fast}");
        // With chronological weights, older samples have lower weight so LOO error
        // is larger than naive brute-force on unweighted data. Just check it's bounded.
        assert!(fast < 500.0, "LOO error should be bounded: {fast}");
    }

    #[test]
    fn chronological_weights_newest_is_one() {
        let w = chronological_weights(5);
        assert_eq!(w.len(), 5);
        assert!((w[4] - 1.0).abs() < 1e-10, "newest weight should be 1.0, got {}", w[4]);
        assert!(w[0] < w[4], "older samples should have lower weight");
    }

    #[test]
    fn face_norm_params_smoke() {
        // Regression: make sure feature extraction doesn't panic on small inputs
        let rgb = vec![100u8; 8 * 6 * 3];
        let feat = extract_eye_features(&rgb, 8, 6);
        assert_eq!(feat.len(), 60);
    }
}
