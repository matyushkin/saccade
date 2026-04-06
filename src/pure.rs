//! PuRe: Pupil Reconstructor.
//!
//! Edge-based pupil detection algorithm from:
//! Santini, Fuhl & Kasneci, "PuRe: Robust pupil detection for real-time
//! pervasive eye tracking", CVIU 2018. arXiv:1712.08900
//!
//! Pipeline:
//! 1. Canny edge detection
//! 2. Edge segment extraction and filtering
//! 3. Segment combination for nearby edges
//! 4. Fitzgibbon ellipse fitting on candidates
//! 5. Confidence scoring: ψ = (ρ + θ + γ) / 3

use crate::edge;
use crate::ellipse::{self, Ellipse};
use crate::frame::Frame;

/// Result of PuRe pupil detection.
#[derive(Debug, Clone, PartialEq)]
pub struct PureResult {
    /// Detected pupil ellipse, if found.
    pub pupil: Option<Ellipse>,
    /// Confidence score ψ in [0.0, 1.0].
    pub confidence: f64,
    /// All candidate ellipses considered (for debugging).
    pub candidates: Vec<PureCandidate>,
}

/// A single candidate ellipse with its confidence breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct PureCandidate {
    pub ellipse: Ellipse,
    /// ρ: aspect ratio score.
    pub rho: f64,
    /// θ: angular spread score.
    pub theta: f64,
    /// γ: outline contrast score.
    pub gamma: f64,
    /// ψ = (ρ + θ + γ) / 3.
    pub psi: f64,
}

/// Configuration for the PuRe detector.
#[derive(Debug, Clone)]
pub struct PureConfig {
    /// Canny low threshold. Default: 15.0.
    pub canny_low: f32,
    /// Canny high threshold. Default: 40.0.
    pub canny_high: f32,
    /// Minimum edge segment length (in pixels). Default: 5.
    pub min_segment_length: usize,
    /// Minimum pupil radius in pixels (at working resolution). Default: 3.0.
    pub min_pupil_radius: f64,
    /// Maximum pupil radius in pixels (at working resolution). Default: 80.0.
    pub max_pupil_radius: f64,
    /// Minimum ellipse aspect ratio (b/a). Default: 0.2.
    pub min_aspect_ratio: f64,
    /// Maximum bounding boxes distance (in pixels) to combine segments. Default: 10.0.
    pub combine_distance: f64,
}

impl Default for PureConfig {
    fn default() -> Self {
        Self {
            canny_low: 15.0,
            canny_high: 40.0,
            min_segment_length: 5,
            min_pupil_radius: 3.0,
            max_pupil_radius: 80.0,
            min_aspect_ratio: 0.2,
            combine_distance: 10.0,
        }
    }
}

/// Detect a pupil in a grayscale eye region using PuRe algorithm.
///
/// Input should be a grayscale eye region (e.g., 320×240 or smaller ROI).
pub fn detect(frame: &dyn Frame, config: &PureConfig) -> PureResult {
    let w = frame.width() as usize;
    let h = frame.height() as usize;

    if w < 10 || h < 10 {
        return PureResult {
            pupil: None,
            confidence: 0.0,
            candidates: Vec::new(),
        };
    }

    // Step 1: Canny edge detection
    let edge_map = edge::canny(frame, config.canny_low, config.canny_high);

    // Step 2: Extract connected edge segments
    let segments = edge::extract_edge_segments(&edge_map, w, h, config.min_segment_length);

    // Step 3: Combine nearby segments
    let combined = combine_segments(&segments, config.combine_distance);

    // Step 4 & 5: Fit ellipses and score candidates
    let mut candidates: Vec<PureCandidate> = Vec::new();

    // Also try fitting all edge pixels as one large segment
    let all_points: Vec<(u32, u32)> = segments.iter().flatten().copied().collect();
    let mut all_segments: Vec<&Vec<(u32, u32)>> = segments.iter().chain(combined.iter()).collect();
    let all_as_vec;
    if all_points.len() >= 6 {
        all_as_vec = all_points;
        all_segments.push(&all_as_vec);
    }

    for segment in all_segments {
        if segment.len() < 6 {
            continue;
        }

        let points: Vec<(f64, f64)> = segment.iter().map(|&(x, y)| (x as f64, y as f64)).collect();

        if let Some(ell) = ellipse::fit_ellipse(&points) {
            // Filter: center must be in frame
            if ell.cx < 0.0 || ell.cx >= w as f64 || ell.cy < 0.0 || ell.cy >= h as f64 {
                continue;
            }

            // Filter: size within pupil bounds
            let radius = (ell.a + ell.b) / 2.0;
            if radius < config.min_pupil_radius || radius > config.max_pupil_radius {
                continue;
            }

            // Filter: aspect ratio
            if ell.aspect_ratio() < config.min_aspect_ratio {
                continue;
            }

            // Compute confidence components
            let rho = ell.aspect_ratio(); // roundness: 1.0 = perfect circle
            let theta = angular_spread(segment);
            let gamma = outline_contrast(frame, &ell);

            let psi = (rho + theta + gamma) / 3.0;

            candidates.push(PureCandidate {
                ellipse: ell,
                rho,
                theta,
                gamma,
                psi,
            });
        }
    }

    // Select best candidate
    let best = candidates.iter().max_by(|a, b| a.psi.partial_cmp(&b.psi).unwrap());

    PureResult {
        pupil: best.map(|c| c.ellipse),
        confidence: best.map(|c| c.psi).unwrap_or(0.0),
        candidates,
    }
}

/// Compute angular spread of edge points around their centroid.
/// Returns a score in [0, 1] where 1 = points spread across all quadrants.
fn angular_spread(segment: &[(u32, u32)]) -> f64 {
    if segment.len() < 2 {
        return 0.0;
    }

    let n = segment.len() as f64;
    let cx: f64 = segment.iter().map(|p| p.0 as f64).sum::<f64>() / n;
    let cy: f64 = segment.iter().map(|p| p.1 as f64).sum::<f64>() / n;

    // Count points in each of 8 angular bins
    let mut bins = [0u32; 8];
    for &(x, y) in segment {
        let angle = (y as f64 - cy).atan2(x as f64 - cx);
        let bin = ((angle + std::f64::consts::PI) / (std::f64::consts::PI / 4.0)) as usize;
        bins[bin.min(7)] += 1;
    }

    // Score: fraction of non-empty bins
    let occupied = bins.iter().filter(|&&b| b > 0).count();
    occupied as f64 / 8.0
}

/// Compute outline contrast: darker inside, brighter outside the ellipse.
/// Returns a score in [0, 1].
fn outline_contrast(frame: &dyn Frame, ell: &Ellipse) -> f64 {
    let pixels = frame.gray_pixels();
    let w = frame.width() as usize;
    let h = frame.height() as usize;

    // Sample points along the ellipse boundary and just inside/outside
    let mut inner_sum = 0.0f64;
    let mut outer_sum = 0.0f64;
    let mut count = 0u32;
    let n_samples = 24;
    let margin = 3.0; // pixels inside/outside

    for i in 0..n_samples {
        let t = 2.0 * std::f64::consts::PI * i as f64 / n_samples as f64;
        let cos_a = ell.angle.cos();
        let sin_a = ell.angle.sin();

        for &(scale, is_inner) in &[(1.0 - margin / ell.a.max(1.0), true), (1.0 + margin / ell.a.max(1.0), false)] {
            let x = ell.a * scale * t.cos();
            let y = ell.b * scale * t.sin();
            let px = (ell.cx + x * cos_a - y * sin_a).round() as i32;
            let py = (ell.cy + x * sin_a + y * cos_a).round() as i32;

            if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                let val = pixels[py as usize * w + px as usize] as f64;
                if is_inner {
                    inner_sum += val;
                } else {
                    outer_sum += val;
                }
                count += 1;
            }
        }
    }

    if count < 4 {
        return 0.0;
    }

    let half = count / 2;
    let inner_mean = if half > 0 { inner_sum / half as f64 } else { 128.0 };
    let outer_mean = if half > 0 { outer_sum / half as f64 } else { 128.0 };

    // Pupil should be darker inside than outside
    let contrast = (outer_mean - inner_mean) / 255.0;
    contrast.clamp(0.0, 1.0)
}

/// Combine nearby segments whose bounding boxes are within `max_dist`.
fn combine_segments(segments: &[Vec<(u32, u32)>], max_dist: f64) -> Vec<Vec<(u32, u32)>> {
    if segments.len() < 2 {
        return Vec::new();
    }

    // Compute bounding boxes
    let bboxes: Vec<(u32, u32, u32, u32)> = segments
        .iter()
        .map(|seg| {
            let min_x = seg.iter().map(|p| p.0).min().unwrap();
            let max_x = seg.iter().map(|p| p.0).max().unwrap();
            let min_y = seg.iter().map(|p| p.1).min().unwrap();
            let max_y = seg.iter().map(|p| p.1).max().unwrap();
            (min_x, min_y, max_x, max_y)
        })
        .collect();

    let mut combined = Vec::new();

    for i in 0..segments.len() {
        for j in (i + 1)..segments.len() {
            let (ax0, ay0, ax1, ay1) = bboxes[i];
            let (bx0, by0, bx1, by1) = bboxes[j];

            // Distance between bounding boxes
            let dx = if ax1 < bx0 {
                (bx0 - ax1) as f64
            } else if bx1 < ax0 {
                (ax0 - bx1) as f64
            } else {
                0.0
            };
            let dy = if ay1 < by0 {
                (by0 - ay1) as f64
            } else if by1 < ay0 {
                (ay0 - by1) as f64
            } else {
                0.0
            };
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= max_dist {
                // Check: one should not fully contain the other
                let a_contains_b =
                    ax0 <= bx0 && ay0 <= by0 && ax1 >= bx1 && ay1 >= by1;
                let b_contains_a =
                    bx0 <= ax0 && by0 <= ay0 && bx1 >= ax1 && by1 >= ay1;

                if !a_contains_b && !b_contains_a {
                    let mut merged = segments[i].clone();
                    merged.extend_from_slice(&segments[j]);
                    combined.push(merged);
                }
            }
        }
    }

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::GrayFrame;

    /// Create a synthetic eye image with a dark elliptical pupil and sharp edge.
    fn make_synthetic_eye(w: u32, h: u32, ell: &Ellipse) -> Vec<u8> {
        let mut data = vec![200u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let cos = ell.angle.cos();
                let sin = ell.angle.sin();
                let dx = x as f64 - ell.cx;
                let dy = y as f64 - ell.cy;
                let xr = dx * cos + dy * sin;
                let yr = -dx * sin + dy * cos;
                let d = ((xr / ell.a).powi(2) + (yr / ell.b).powi(2)).sqrt();
                if d < 0.9 {
                    data[(y * w + x) as usize] = 30;
                } else if d < 1.1 {
                    // Sharp transition at boundary (2px wide)
                    let t = (d - 0.9) / 0.2;
                    data[(y * w + x) as usize] = (30.0 + 170.0 * t) as u8;
                }
            }
        }
        data
    }

    #[test]
    fn detects_synthetic_pupil() {
        // Larger image with bigger pupil for reliable Canny edge detection
        let expected = Ellipse {
            cx: 160.0, cy: 120.0, a: 40.0, b: 30.0, angle: 0.0,
        };
        let (w, h) = (320, 240);
        let data = make_synthetic_eye(w, h, &expected);
        let frame = GrayFrame::new(w, h, &data);
        let config = PureConfig {
            canny_low: 10.0,
            canny_high: 30.0,
            min_segment_length: 5,
            combine_distance: 15.0,
            ..PureConfig::default()
        };
        let result = detect(&frame, &config);

        assert!(
            result.pupil.is_some(),
            "should detect a pupil, got {} candidates",
            result.candidates.len()
        );
        if let Some(pupil) = result.pupil {
            let error = ((pupil.cx - expected.cx).powi(2) + (pupil.cy - expected.cy).powi(2)).sqrt();
            assert!(
                error < 15.0,
                "center error {error:.1}px too large (detected: ({:.1}, {:.1}), expected: ({}, {}))",
                pupil.cx, pupil.cy, expected.cx, expected.cy
            );
        }
    }

    #[test]
    fn empty_frame_returns_none() {
        let data = vec![128u8; 160 * 120];
        let frame = GrayFrame::new(160, 120, &data);
        let result = detect(&frame, &PureConfig::default());
        assert!(result.pupil.is_none());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn small_frame_returns_none() {
        let data = vec![128u8; 5 * 5];
        let frame = GrayFrame::new(5, 5, &data);
        let result = detect(&frame, &PureConfig::default());
        assert!(result.pupil.is_none());
    }
}
