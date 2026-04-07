//! Ridge regression with WebGazer-style eye feature extraction.
//!
//! Port of WebGazer.js's core algorithm:
//! - Extract eye patch as 10×6 grayscale, histogram equalized (60-D per eye)
//! - Concatenate both eyes → 120-D feature vector
//! - Ridge regression: β = (XᵀX + λI)⁻¹ Xᵀy
//! - Retrain on every prediction using circular buffer of last N samples
//!
//! Reference: Papoutsaki et al., "WebGazer: Scalable Webcam Eye Tracking
//! Using User Interactions", IJCAI 2016.

use nalgebra::{DMatrix, DVector};

/// Standard WebGazer eye patch size.
pub const EYE_PATCH_W: usize = 10;
pub const EYE_PATCH_H: usize = 6;
pub const EYE_FEAT_LEN: usize = EYE_PATCH_W * EYE_PATCH_H; // 60 per eye
pub const BOTH_EYES_FEAT_LEN: usize = EYE_FEAT_LEN * 2; // 120 total

/// Extract WebGazer-style features from an RGB eye patch.
/// Returns a 60-element feature vector (10×6 histogram-equalized grayscale).
pub fn extract_eye_features(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    if rgb.len() != width * height * 3 || width == 0 || height == 0 {
        return vec![0.0; EYE_FEAT_LEN];
    }

    // Step 1: Bilinear resize to 10×6 (preserving RGB)
    let resized = bilinear_resize_rgb(rgb, width, height, EYE_PATCH_W, EYE_PATCH_H);

    // Step 2: Convert to grayscale (ITU-R BT.601)
    let mut gray = vec![0u8; EYE_PATCH_W * EYE_PATCH_H];
    for i in 0..EYE_PATCH_W * EYE_PATCH_H {
        let r = resized[i * 3] as f32;
        let g = resized[i * 3 + 1] as f32;
        let b = resized[i * 3 + 2] as f32;
        gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
    }

    // Step 3: Histogram equalization (WebGazer uses sampling step=5)
    histogram_equalize(&gray, 5)
}

/// Extract features from a grayscale eye patch (when we only have grayscale data).
pub fn extract_eye_features_gray(gray_patch: &[u8], width: usize, height: usize) -> Vec<f32> {
    if gray_patch.len() != width * height || width == 0 || height == 0 {
        return vec![0.0; EYE_FEAT_LEN];
    }

    // Bilinear resize grayscale to 10×6
    let mut resized = vec![0u8; EYE_PATCH_W * EYE_PATCH_H];
    for dy in 0..EYE_PATCH_H {
        for dx in 0..EYE_PATCH_W {
            let sx = dx as f32 * width as f32 / EYE_PATCH_W as f32;
            let sy = dy as f32 * height as f32 / EYE_PATCH_H as f32;
            let x0 = (sx as usize).min(width - 1);
            let y0 = (sy as usize).min(height - 1);
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let p00 = gray_patch[y0 * width + x0] as f32;
            let p01 = gray_patch[y0 * width + x1] as f32;
            let p10 = gray_patch[y1 * width + x0] as f32;
            let p11 = gray_patch[y1 * width + x1] as f32;
            let v = p00 * (1.0 - fx) * (1.0 - fy)
                + p01 * fx * (1.0 - fy)
                + p10 * (1.0 - fx) * fy
                + p11 * fx * fy;
            resized[dy * EYE_PATCH_W + dx] = v as u8;
        }
    }

    histogram_equalize(&resized, 5)
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

/// WebGazer-compatible histogram equalization.
/// `step` controls histogram sampling (WebGazer uses step=5).
fn histogram_equalize(src: &[u8], step: usize) -> Vec<f32> {
    let len = src.len();
    let mut hist = [0u32; 256];
    let mut i = 0;
    let mut sample_count = 0u32;
    while i < len {
        hist[src[i] as usize] += 1;
        sample_count += 1;
        i += step;
    }

    // Integral histogram (CDF), normalized to [0, 255]
    let mut cdf = [0.0f32; 256];
    let mut acc = 0u32;
    let scale = 255.0 / sample_count as f32;
    for i in 0..256 {
        acc += hist[i];
        cdf[i] = acc as f32 * scale;
    }

    src.iter().map(|&p| cdf[p as usize]).collect()
}

/// A single calibration sample: features + target screen position.
#[derive(Debug, Clone)]
pub struct RidgeSample {
    pub features: Vec<f32>,
    pub target_x: f32,
    pub target_y: f32,
}

/// Ridge regressor with circular buffer.
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

    /// Solve ridge regression and predict for given features.
    /// Returns None if not enough samples or solver fails.
    pub fn predict(&self, features: &[f32]) -> Option<(f32, f32)> {
        if self.samples.len() < 3 || features.len() != self.feat_len {
            return None;
        }

        let n = self.samples.len();
        let p = self.feat_len;

        // Build design matrix X (n × p) and target vectors
        let mut x_data = Vec::with_capacity(n * p);
        let mut y_x = Vec::with_capacity(n);
        let mut y_y = Vec::with_capacity(n);
        for s in &self.samples {
            for &f in &s.features {
                x_data.push(f as f64);
            }
            y_x.push(s.target_x as f64);
            y_y.push(s.target_y as f64);
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

        // Solve (XᵀX + λI) β = Xᵀy using LU decomposition
        let decomp = xtx.lu();
        let beta_x = decomp.solve(&xty_x)?;
        let beta_y = decomp.solve(&xty_y)?;

        // Predict: f · β
        let mut px = 0.0f64;
        let mut py = 0.0f64;
        for i in 0..p {
            px += features[i] as f64 * beta_x[i];
            py += features[i] as f64 * beta_y[i];
        }

        Some((px as f32, py as f32))
    }
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
    fn ridge_fits_linear_data() {
        // target_x = 2 * f[0] + 1 (linear in one feature)
        let mut reg = RidgeRegressor::new(20, 1e-5, 3);
        for i in 0..10 {
            let features = vec![i as f32, 0.0, 0.0];
            reg.add_sample(features, 2.0 * i as f32 + 1.0, 0.0);
        }
        // Predict for x=5 → expect ~11
        let pred = reg.predict(&[5.0, 0.0, 0.0]).unwrap();
        assert!((pred.0 - 11.0).abs() < 1.0, "predicted {}", pred.0);
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
    fn histogram_equalize_preserves_length() {
        let src = vec![10u8, 50, 100, 150, 200, 250, 50, 100];
        let out = histogram_equalize(&src, 1);
        assert_eq!(out.len(), src.len());
    }
}
