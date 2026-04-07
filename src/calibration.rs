//! Gaze calibration and head-pose-invariant coordinate system.
//!
//! Calibration pipeline:
//! 1. User looks at known screen points during calibration
//! 2. For each point, record normalized pupil position (relative to eye ROI, de-rotated)
//! 3. Fit polynomial mapping: normalized pupil coords → screen coords
//! 4. At runtime: detect pupils → normalize → de-rotate → map to screen

/// Normalized, head-pose-invariant pupil position in [-1, 1].
#[derive(Debug, Clone, Copy)]
pub struct NormalizedPupil {
    /// Horizontal position: -1 = looking far left, +1 = far right.
    pub x: f64,
    /// Vertical position: -1 = looking up, +1 = looking down.
    pub y: f64,
}

/// Compute normalized pupil position from raw detections.
///
/// - `pupil`: detected pupil center (absolute pixels)
/// - `roi_center`: center of the eye ROI (absolute pixels)
/// - `roi_size`: (width, height) of eye ROI
/// - `roll_angle`: head roll angle in radians (from inter-pupil line)
pub fn normalize_pupil(
    pupil: (f64, f64),
    roi_center: (f64, f64),
    roi_size: (f64, f64),
    roll_angle: f64,
) -> NormalizedPupil {
    // Relative to ROI center, normalized to [-1, 1]
    let nx = if roi_size.0 > 1.0 {
        (pupil.0 - roi_center.0) / (roi_size.0 / 2.0)
    } else {
        0.0
    };
    let ny = if roi_size.1 > 1.0 {
        (pupil.1 - roi_center.1) / (roi_size.1 / 2.0)
    } else {
        0.0
    };

    // De-rotate by head roll
    let cos = roll_angle.cos();
    let sin = roll_angle.sin();
    NormalizedPupil {
        x: (nx * cos + ny * sin).clamp(-1.5, 1.5),
        y: (-nx * sin + ny * cos).clamp(-1.5, 1.5),
    }
}

/// Compute head roll angle from two pupil positions.
/// Returns angle in radians (positive = clockwise tilt).
pub fn head_roll(left_pupil: (f64, f64), right_pupil: (f64, f64)) -> f64 {
    let dy = right_pupil.1 - left_pupil.1;
    let dx = right_pupil.0 - left_pupil.0;
    dy.atan2(dx)
}

/// PPERV (Pupil Position in Eye Reference Vector) — head-pose-invariant feature.
///
/// Computes pupil position relative to eye corners, normalized by eye width
/// and de-rotated to the eye's local coordinate frame. This is more stable
/// than ROI-bounding-box normalization because eye corners are well-defined
/// landmarks that don't move when the eye rotates.
///
/// - `pupil`: detected pupil center (absolute image pixels)
/// - `outer_corner`, `inner_corner`: eye corner landmarks (absolute pixels)
pub fn pperv(
    pupil: (f64, f64),
    outer_corner: (f64, f64),
    inner_corner: (f64, f64),
) -> NormalizedPupil {
    let cx = (outer_corner.0 + inner_corner.0) / 2.0;
    let cy = (outer_corner.1 + inner_corner.1) / 2.0;
    let dx = inner_corner.0 - outer_corner.0;
    let dy = inner_corner.1 - outer_corner.1;
    let eye_width = (dx * dx + dy * dy).sqrt();
    if eye_width < 1.0 {
        return NormalizedPupil { x: 0.0, y: 0.0 };
    }

    // Eye axis angle (from outer to inner corner)
    let angle = dy.atan2(dx);
    let cos = angle.cos();
    let sin = angle.sin();

    // Translate pupil to eye center
    let rel_x = pupil.0 - cx;
    let rel_y = pupil.1 - cy;

    // Rotate into eye's local frame (x = along eye axis, y = perpendicular)
    let local_x = rel_x * cos + rel_y * sin;
    let local_y = -rel_x * sin + rel_y * cos;

    // Normalize by eye width (not 0.5 × width, so output is in [-0.5, 0.5] for extremes)
    NormalizedPupil {
        x: (local_x / eye_width).clamp(-1.5, 1.5),
        y: (local_y / eye_width).clamp(-1.5, 1.5),
    }
}

/// A single calibration sample: normalized pupil → screen point.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationSample {
    pub pupil: NormalizedPupil,
    pub screen_x: f64,
    pub screen_y: f64,
}

/// Blink/eye-close calibration data.
#[derive(Debug, Clone, Copy)]
pub struct BlinkCalibration {
    /// Dark ratio when eyes are open (per eye).
    pub open_dark_ratio: f64,
    /// Dark ratio when eyes are closed (per eye).
    pub closed_dark_ratio: f64,
    /// Threshold: midpoint between open and closed.
    pub threshold: f64,
}

impl BlinkCalibration {
    pub fn from_samples(open: f64, closed: f64) -> Self {
        Self {
            open_dark_ratio: open,
            closed_dark_ratio: closed,
            threshold: (open + closed) / 2.0,
        }
    }

    pub fn is_closed(&self, dark_ratio: f64) -> bool {
        dark_ratio < self.threshold
    }
}

/// Calibrated gaze mapper using 2nd-degree polynomial regression.
///
/// Maps normalized pupil (x, y) → screen (x, y) using:
/// ```text
/// screen_x = a0 + a1*px + a2*py + a3*px² + a4*py² + a5*px*py
/// screen_y = b0 + b1*px + b2*py + b3*px² + b4*py² + b5*px*py
/// ```
#[derive(Debug, Clone)]
pub struct GazeMapper {
    /// Coefficients for screen X: [a0, a1, a2, a3, a4, a5]
    pub coeffs_x: [f64; 6],
    /// Coefficients for screen Y: [b0, b1, b2, b3, b4, b5]
    pub coeffs_y: [f64; 6],
    /// Whether the mapper has been calibrated.
    pub calibrated: bool,
}

impl GazeMapper {
    pub fn new() -> Self {
        Self {
            coeffs_x: [0.0; 6],
            coeffs_y: [0.0; 6],
            calibrated: false,
        }
    }

    /// Calibrate from a set of samples.
    /// Requires at least 6 samples for a unique solution.
    /// More samples → least-squares fit.
    pub fn calibrate(&mut self, samples: &[CalibrationSample]) -> bool {
        if samples.len() < 3 {
            return false;
        }

        let n = samples.len();

        // Build design matrix A (n × 6) and target vectors bx, by
        // Each row: [1, px, py, px², py², px*py]
        let mut a_data = vec![0.0f64; n * 6];
        let mut bx = vec![0.0f64; n];
        let mut by = vec![0.0f64; n];

        for (i, s) in samples.iter().enumerate() {
            let px = s.pupil.x;
            let py = s.pupil.y;
            a_data[i * 6] = 1.0;
            a_data[i * 6 + 1] = px;
            a_data[i * 6 + 2] = py;
            a_data[i * 6 + 3] = px * px;
            a_data[i * 6 + 4] = py * py;
            a_data[i * 6 + 5] = px * py;
            bx[i] = s.screen_x;
            by[i] = s.screen_y;
        }

        // Solve via normal equations: (AᵀA)c = Aᵀb
        // AᵀA is 6×6, Aᵀb is 6×1
        let ncols = 6.min(n); // can't have more coefficients than samples
        let mut ata = vec![0.0f64; ncols * ncols];
        let mut atbx = vec![0.0f64; ncols];
        let mut atby = vec![0.0f64; ncols];

        for i in 0..ncols {
            for j in 0..ncols {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a_data[k * 6 + i] * a_data[k * 6 + j];
                }
                ata[i * ncols + j] = sum;
            }
            let mut sx = 0.0;
            let mut sy = 0.0;
            for k in 0..n {
                sx += a_data[k * 6 + i] * bx[k];
                sy += a_data[k * 6 + i] * by[k];
            }
            atbx[i] = sx;
            atby[i] = sy;
        }

        // Solve using Gaussian elimination
        if let (Some(cx), Some(cy)) = (
            solve_linear(&ata, &atbx, ncols),
            solve_linear(&ata, &atby, ncols),
        ) {
            for i in 0..ncols {
                self.coeffs_x[i] = cx[i];
                self.coeffs_y[i] = cy[i];
            }
            self.calibrated = true;
            true
        } else {
            false
        }
    }

    /// Map normalized pupil position to screen coordinates.
    pub fn map(&self, pupil: &NormalizedPupil) -> (f64, f64) {
        let px = pupil.x;
        let py = pupil.y;
        let features = [1.0, px, py, px * px, py * py, px * py];
        let sx: f64 = self.coeffs_x.iter().zip(&features).map(|(c, f)| c * f).sum();
        let sy: f64 = self.coeffs_y.iter().zip(&features).map(|(c, f)| c * f).sum();
        (sx, sy)
    }
}

impl Default for GazeMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Augmented matrix [A|b]
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None; // singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / aug[col * (n + 1) + col];
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn normalize_pupil_centered() {
        let p = normalize_pupil((100.0, 50.0), (100.0, 50.0), (80.0, 40.0), 0.0);
        assert!((p.x).abs() < 0.01);
        assert!((p.y).abs() < 0.01);
    }

    #[test]
    fn normalize_pupil_right() {
        let p = normalize_pupil((140.0, 50.0), (100.0, 50.0), (80.0, 40.0), 0.0);
        assert!((p.x - 1.0).abs() < 0.01);
        assert!((p.y).abs() < 0.01);
    }

    #[test]
    fn normalize_with_roll_compensation() {
        // Without roll: pupil at (140, 50), center (100,50), size (80,80) → nx=1.0, ny=0.0
        let p0 = normalize_pupil((140.0, 50.0), (100.0, 50.0), (80.0, 80.0), 0.0);
        assert!((p0.x - 1.0).abs() < 0.01, "p0.x={}", p0.x);
        assert!(p0.y.abs() < 0.01, "p0.y={}", p0.y);

        // With 30° roll: same gaze direction but head tilted
        // The de-rotated result should still show rightward gaze
        let roll = PI / 6.0; // 30 degrees
        // Pupil position rotated by 30° around center: (134.6, 70.0)
        let cos = roll.cos();
        let sin = roll.sin();
        let dx = 40.0; // original offset (140-100)
        let rpx = 100.0 + dx * cos;
        let rpy = 50.0 + dx * sin;
        let p = normalize_pupil((rpx, rpy), (100.0, 50.0), (80.0, 80.0), roll);
        assert!((p.x - 1.0).abs() < 0.1, "x={}", p.x);
        assert!(p.y.abs() < 0.1, "y={}", p.y);
    }

    #[test]
    fn pperv_pupil_at_center() {
        // Eye corners at (100, 50) and (140, 50), pupil at center (120, 50)
        let p = pperv((120.0, 50.0), (100.0, 50.0), (140.0, 50.0));
        assert!(p.x.abs() < 0.01, "x={}", p.x);
        assert!(p.y.abs() < 0.01, "y={}", p.y);
    }

    #[test]
    fn pperv_pupil_right_of_center() {
        // Pupil moved 10px right (25% of eye width)
        let p = pperv((130.0, 50.0), (100.0, 50.0), (140.0, 50.0));
        assert!((p.x - 0.25).abs() < 0.01, "x={}", p.x);
    }

    #[test]
    fn pperv_rotation_invariant() {
        // Same gaze (pupil at center), different head rotations
        let p1 = pperv((120.0, 50.0), (100.0, 50.0), (140.0, 50.0));
        // Rotate 45° — pupil and corners rotate together
        let cx = 120.0; let cy = 50.0;
        let rot = |x: f64, y: f64, ang: f64| {
            let dx = x - cx; let dy = y - cy;
            (cx + dx * ang.cos() - dy * ang.sin(), cy + dx * ang.sin() + dy * ang.cos())
        };
        let ang = std::f64::consts::PI / 4.0;
        let pupil_r = rot(120.0, 50.0, ang);
        let c1_r = rot(100.0, 50.0, ang);
        let c2_r = rot(140.0, 50.0, ang);
        let p2 = pperv(pupil_r, c1_r, c2_r);
        // Should give the same normalized coordinates
        assert!((p1.x - p2.x).abs() < 0.01);
        assert!((p1.y - p2.y).abs() < 0.01);
    }

    #[test]
    fn head_roll_flat() {
        let angle = head_roll((100.0, 50.0), (200.0, 50.0));
        assert!(angle.abs() < 0.01);
    }

    #[test]
    fn head_roll_tilted() {
        let angle = head_roll((100.0, 50.0), (200.0, 100.0));
        assert!((angle - (50.0f64 / 100.0).atan()).abs() < 0.01);
    }

    #[test]
    fn calibrate_linear_mapping() {
        let mut mapper = GazeMapper::new();

        // 9-point grid calibration
        let samples: Vec<CalibrationSample> = vec![
            (-1.0, -1.0, 0.0, 0.0),
            (0.0, -1.0, 960.0, 0.0),
            (1.0, -1.0, 1920.0, 0.0),
            (-1.0, 0.0, 0.0, 540.0),
            (0.0, 0.0, 960.0, 540.0),
            (1.0, 0.0, 1920.0, 540.0),
            (-1.0, 1.0, 0.0, 1080.0),
            (0.0, 1.0, 960.0, 1080.0),
            (1.0, 1.0, 1920.0, 1080.0),
        ]
        .into_iter()
        .map(|(px, py, sx, sy)| CalibrationSample {
            pupil: NormalizedPupil { x: px, y: py },
            screen_x: sx,
            screen_y: sy,
        })
        .collect();

        assert!(mapper.calibrate(&samples));
        assert!(mapper.calibrated);

        // Test center
        let (sx, sy) = mapper.map(&NormalizedPupil { x: 0.0, y: 0.0 });
        assert!((sx - 960.0).abs() < 1.0, "center sx={sx}");
        assert!((sy - 540.0).abs() < 1.0, "center sy={sy}");

        // Test corner
        let (sx, sy) = mapper.map(&NormalizedPupil { x: 1.0, y: 1.0 });
        assert!((sx - 1920.0).abs() < 1.0, "corner sx={sx}");
        assert!((sy - 1080.0).abs() < 1.0, "corner sy={sy}");

        // Test interpolation (midpoint)
        let (sx, sy) = mapper.map(&NormalizedPupil { x: 0.5, y: 0.5 });
        assert!((sx - 1440.0).abs() < 50.0, "mid sx={sx}");
        assert!((sy - 810.0).abs() < 50.0, "mid sy={sy}");
    }

    #[test]
    fn blink_calibration() {
        let bc = BlinkCalibration::from_samples(0.12, 0.03);
        assert!(!bc.is_closed(0.10)); // open
        assert!(bc.is_closed(0.05)); // closed
        assert!((bc.threshold - 0.075).abs() < 0.001);
    }
}
