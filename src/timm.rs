//! Timm & Barth gradient-based eye center localization.
//!
//! Implements the algorithm from:
//! Timm & Barth, "Accurate Eye Centre Localisation by Means of Gradients", VISAPP 2011.
//!
//! For each candidate center point `c`, we maximize:
//! ```text
//! objective(c) = Σ_x w(x) · max(0, d(x,c)ᵀ · g(x))²
//! ```
//! where `d(x,c)` is the normalized displacement, `g(x)` is the image gradient,
//! and `w(x)` is a darkness weight (inverted, blurred image).

use crate::frame::Frame;

/// Result of Timm & Barth eye center detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientCenter {
    /// X coordinate of detected center in the input frame.
    pub x: f64,
    /// Y coordinate of detected center in the input frame.
    pub y: f64,
    /// Normalized objective value in [0.0, 1.0]. Higher = more confident.
    pub confidence: f64,
}

/// Configuration for the Timm & Barth detector.
#[derive(Debug, Clone)]
pub struct TimmConfig {
    /// Gradient magnitude threshold. Pixels with gradient magnitude below
    /// this fraction of the max gradient are skipped. Default: 0.3.
    pub gradient_threshold: f64,
    /// Whether to use a weight map (inverted image) to bias toward dark regions.
    /// Default: true.
    pub use_weight_map: bool,
    /// Sigma for Gaussian blur on weight map (in pixels). 0 = no blur.
    /// Default: 3.0.
    pub weight_blur_sigma: f64,
}

impl Default for TimmConfig {
    fn default() -> Self {
        Self {
            gradient_threshold: 0.3,
            use_weight_map: true,
            weight_blur_sigma: 3.0,
        }
    }
}

/// Detect the eye center in a grayscale eye region using gradient analysis.
///
/// Input should be a cropped eye region (e.g., 40×30 to 160×120).
/// Smaller inputs are faster. For sub-ms performance, use ~40×30.
///
/// For larger inputs (>80px wide), uses coarse-to-fine search:
/// first scans every 4th pixel, then refines in a 7×7 neighborhood.
pub fn detect_center(frame: &dyn Frame, config: &TimmConfig) -> GradientCenter {
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let pixels = frame.gray_pixels();

    if w < 3 || h < 3 {
        return GradientCenter {
            x: w as f64 / 2.0,
            y: h as f64 / 2.0,
            confidence: 0.0,
        };
    }

    // Compute gradients (Sobel-like: simple central differences)
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    let mut mag = vec![0.0f32; w * h];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let dx = pixels[idx + 1] as f32 - pixels[idx - 1] as f32;
            let dy = pixels[idx + w] as f32 - pixels[idx - w] as f32;
            let m = (dx * dx + dy * dy).sqrt();
            gx[idx] = dx;
            gy[idx] = dy;
            mag[idx] = m;
        }
    }

    // Find max gradient magnitude for thresholding
    let max_mag = mag.iter().cloned().fold(0.0f32, f32::max);
    let mag_threshold = max_mag * config.gradient_threshold as f32;

    // Compute weight map (inverted image, optionally blurred)
    let weights: Vec<f32> = if config.use_weight_map {
        let inverted: Vec<f32> = pixels.iter().map(|&p| 255.0 - p as f32).collect();
        if config.weight_blur_sigma > 0.0 {
            box_blur(&inverted, w, h, config.weight_blur_sigma as f32)
        } else {
            inverted
        }
    } else {
        vec![1.0; w * h]
    };

    // Accumulator-based approach: O(G × W × H) instead of O(W² × H²).
    //
    // For each gradient point p, we compute its weighted contribution to every
    // candidate center c:  w(p) · max(0, d(p,c)·g(p))²
    //
    // This inverts the loop order: outer = gradient points, inner = candidate centers.
    // Each gradient point "votes" for all centers along its gradient direction.
    let mut accum = vec![0.0f32; w * h];

    // Pre-collect gradient points to avoid branch in hot loop
    let mut grad_points: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
    for py in 1..h - 1 {
        for px in 1..w - 1 {
            let pidx = py * w + px;
            let m = mag[pidx];
            if m < mag_threshold {
                continue;
            }
            let gx_n = gx[pidx] / m;
            let gy_n = gy[pidx] / m;
            let wt = weights[pidx];
            grad_points.push((px, py, gx_n, gy_n, wt));
        }
    }

    // Coarse-to-fine: use step>1 for large frames, then refine
    let coarse_step = if w > 80 || h > 80 { 4 } else { 1 };

    // Coarse pass: use subsampled gradient points for speed
    let coarse_grads: Vec<_> = if coarse_step > 1 {
        grad_points.iter().step_by(coarse_step).copied().collect()
    } else {
        grad_points.clone()
    };
    accumulate_votes(&coarse_grads, &mut accum, w, h, coarse_step);

    // Find coarse best
    let mut best_val = 0.0f32;
    let mut best_x = w / 2;
    let mut best_y = h / 2;

    for cy in (1..h - 1).step_by(coarse_step) {
        for cx in (1..w - 1).step_by(coarse_step) {
            let val = accum[cy * w + cx];
            if val > best_val {
                best_val = val;
                best_x = cx;
                best_y = cy;
            }
        }
    }

    // Fine pass: refine in neighborhood around coarse best
    if coarse_step > 1 {
        let margin = coarse_step + 3;
        let fy0 = best_y.saturating_sub(margin).max(1);
        let fy1 = (best_y + margin + 1).min(h - 1);
        let fx0 = best_x.saturating_sub(margin).max(1);
        let fx1 = (best_x + margin + 1).min(w - 1);

        // Re-accumulate at pixel resolution in the neighborhood
        let mut fine_accum = vec![0.0f32; w * h];
        for &(px, py, gx_n, gy_n, wt) in &grad_points {
            let px_f = px as f32;
            let py_f = py as f32;
            for cy in fy0..fy1 {
                let dy = py_f - cy as f32;
                let dy_g = dy * gy_n;
                for cx in fx0..fx1 {
                    let dx = px_f - cx as f32;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < 1.0 { continue; }
                    let dot_unnorm = dx * gx_n + dy_g;
                    if dot_unnorm <= 0.0 { continue; }
                    fine_accum[cy * w + cx] += wt * dot_unnorm * dot_unnorm / dist_sq;
                }
            }
        }

        for cy in fy0..fy1 {
            for cx in fx0..fx1 {
                let val = fine_accum[cy * w + cx];
                if val > best_val {
                    best_val = val;
                    best_x = cx;
                    best_y = cy;
                }
            }
        }
    }

    // Normalize confidence to [0, 1]
    let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
    let num_gradient_pixels = grad_points.len() as f64;
    let max_possible = num_gradient_pixels * max_weight as f64;
    let confidence = if max_possible > 0.0 {
        (best_val as f64 / max_possible).clamp(0.0, 1.0)
    } else {
        0.0
    };

    GradientCenter {
        x: best_x as f64,
        y: best_y as f64,
        confidence,
    }
}

/// Accumulate gradient votes into the accumulator map with given step size.
fn accumulate_votes(
    grad_points: &[(usize, usize, f32, f32, f32)],
    accum: &mut [f32],
    w: usize,
    h: usize,
    step: usize,
) {
    for &(px, py, gx_n, gy_n, wt) in grad_points {
        let px_f = px as f32;
        let py_f = py as f32;

        for cy in (1..h - 1).step_by(step) {
            let dy = py_f - cy as f32;
            let dy_g = dy * gy_n;

            for cx in (1..w - 1).step_by(step) {
                let dx = px_f - cx as f32;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < 1.0 {
                    continue;
                }
                let dot_unnorm = dx * gx_n + dy_g;
                if dot_unnorm <= 0.0 {
                    continue;
                }
                accum[cy * w + cx] += wt * dot_unnorm * dot_unnorm / dist_sq;
            }
        }
    }
}

/// Simple box blur approximation of Gaussian blur.
/// Applies 3 passes of box blur (approximates Gaussian well).
fn box_blur(input: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    // Box blur radius from sigma (3 passes approximation)
    let radius = ((sigma * 2.5).round() as usize / 3).max(1);
    let mut buf_a = input.to_vec();
    let mut buf_b = vec![0.0f32; w * h];

    for _ in 0..3 {
        // Horizontal pass
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius + 1).min(w);
                for bx in x_start..x_end {
                    sum += buf_a[y * w + bx];
                    count += 1;
                }
                buf_b[y * w + x] = sum / count as f32;
            }
        }
        // Vertical pass
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius + 1).min(h);
                for by in y_start..y_end {
                    sum += buf_b[by * w + x];
                    count += 1;
                }
                buf_a[y * w + x] = sum / count as f32;
            }
        }
    }
    buf_a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::GrayFrame;

    /// Create a synthetic eye-like image: dark circle (pupil) on lighter background.
    fn make_synthetic_eye(w: u32, h: u32, cx: u32, cy: u32, radius: u32) -> Vec<u8> {
        let mut data = vec![180u8; (w * h) as usize]; // light background
        for y in 0..h {
            for x in 0..w {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let dist = ((dx * dx + dy * dy) as f64).sqrt();
                if dist < radius as f64 {
                    // Dark pupil with smooth edge
                    let edge_factor = (dist / radius as f64).clamp(0.0, 1.0);
                    data[(y * w + x) as usize] = (30.0 + 150.0 * edge_factor * edge_factor) as u8;
                }
            }
        }
        data
    }

    #[test]
    fn detects_center_of_synthetic_pupil() {
        let (w, h) = (40, 30);
        let (true_cx, true_cy) = (20, 15);
        let data = make_synthetic_eye(w, h, true_cx, true_cy, 8);
        let frame = GrayFrame::new(w, h, &data);
        let result = detect_center(&frame, &TimmConfig::default());

        let error = ((result.x - true_cx as f64).powi(2) + (result.y - true_cy as f64).powi(2)).sqrt();
        assert!(
            error < 3.0,
            "detection error {error:.1}px too large (detected: ({:.1}, {:.1}), true: ({true_cx}, {true_cy}))",
            result.x, result.y
        );
        assert!(result.confidence > 0.0, "confidence should be positive");
    }

    #[test]
    fn detects_off_center_pupil() {
        let (w, h) = (60, 40);
        let (true_cx, true_cy) = (15, 25);
        let data = make_synthetic_eye(w, h, true_cx, true_cy, 7);
        let frame = GrayFrame::new(w, h, &data);
        let result = detect_center(&frame, &TimmConfig::default());

        let error = ((result.x - true_cx as f64).powi(2) + (result.y - true_cy as f64).powi(2)).sqrt();
        assert!(
            error < 4.0,
            "detection error {error:.1}px (detected: ({:.1}, {:.1}), true: ({true_cx}, {true_cy}))",
            result.x, result.y
        );
    }

    #[test]
    fn tiny_frame_returns_center() {
        let data = vec![100u8; 4];
        let frame = GrayFrame::new(2, 2, &data);
        let result = detect_center(&frame, &TimmConfig::default());
        assert_eq!(result.confidence, 0.0);
    }
}
