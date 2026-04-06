//! Ellipse fitting and representation.
//!
//! Implements the Fitzgibbon direct least-squares ellipse fitting algorithm:
//! Fitzgibbon, Pilu & Fisher, "Direct Least Square Fitting of Ellipses", PAMI 1999.

use nalgebra::DMatrix;

/// An ellipse parameterized by center, semi-axes, and rotation angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    /// Center X coordinate.
    pub cx: f64,
    /// Center Y coordinate.
    pub cy: f64,
    /// Semi-major axis length.
    pub a: f64,
    /// Semi-minor axis length.
    pub b: f64,
    /// Rotation angle in radians (from positive X axis, counter-clockwise).
    pub angle: f64,
}

impl Ellipse {
    /// Aspect ratio (b/a), always in [0, 1].
    pub fn aspect_ratio(&self) -> f64 {
        if self.a > self.b {
            self.b / self.a
        } else {
            self.a / self.b
        }
    }

    /// Area of the ellipse.
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.a * self.b
    }

    /// Check if a point (x, y) is inside the ellipse.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let cos = self.angle.cos();
        let sin = self.angle.sin();
        let dx = x - self.cx;
        let dy = y - self.cy;
        let xr = dx * cos + dy * sin;
        let yr = -dx * sin + dy * cos;
        let a = self.a.max(self.b);
        let b = self.a.min(self.b);
        (xr * xr) / (a * a) + (yr * yr) / (b * b) <= 1.0
    }
}

/// Fit an ellipse to a set of 2D points using Fitzgibbon's direct least-squares method.
///
/// Requires at least 6 points. Returns `None` if fitting fails (degenerate configuration,
/// not enough points, or result is not an ellipse).
pub fn fit_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 6 {
        return None;
    }

    let n = points.len();

    // Build design matrix D (n×6): each row is [x², xy, y², x, y, 1]
    let mut d = DMatrix::<f64>::zeros(n, 6);
    for (i, &(x, y)) in points.iter().enumerate() {
        d[(i, 0)] = x * x;
        d[(i, 1)] = x * y;
        d[(i, 2)] = y * y;
        d[(i, 3)] = x;
        d[(i, 4)] = y;
        d[(i, 5)] = 1.0;
    }

    // Constraint matrix C (6×6): enforces 4ac - b² > 0
    let mut c_mat = DMatrix::<f64>::zeros(6, 6);
    c_mat[(0, 2)] = 2.0;
    c_mat[(1, 1)] = -1.0;
    c_mat[(2, 0)] = 2.0;

    // Scatter matrix S = DᵀD
    let s = d.transpose() * &d;

    // Solve generalized eigenvalue problem: S·a = λ·C·a
    // Equivalent to: S⁻¹·C·a = (1/λ)·a
    // But S may be ill-conditioned, so use the partitioned approach with SVD fallback.

    // Partition S into 3×3 blocks
    let s11 = s.view((0, 0), (3, 3)).clone_owned();
    let s12 = s.view((0, 3), (3, 3)).clone_owned();
    let s21 = s.view((3, 0), (3, 3)).clone_owned();
    let s22 = s.view((3, 3), (3, 3)).clone_owned();

    // S22 inverse
    let s22_inv = s22.clone().try_inverse()?;

    // Reduced matrix: M = C1⁻¹ (S11 - S12 S22⁻¹ S21)
    let t_mat = s11 - &s12 * &s22_inv * &s21;

    // C1⁻¹ = [[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]]
    let mut c1_inv = DMatrix::<f64>::zeros(3, 3);
    c1_inv[(0, 2)] = 0.5;
    c1_inv[(1, 1)] = -1.0;
    c1_inv[(2, 0)] = 0.5;

    let mc = &c1_inv * &t_mat;

    // Find eigenvalues/eigenvectors via power iteration on shifted matrices.
    // We need the eigenvector where 4ac - b² > 0.
    // Try all 3 eigenvectors via shifted inverse iteration with multiple shifts.
    let eigvecs = find_eigenvectors_3x3(&mc);

    let mut best_conic: Option<[f64; 6]> = None;
    let mut best_constraint = 0.0f64;

    for eigvec in &eigvecs {
        let (a1_0, a1_1, a1_2) = (eigvec[0], eigvec[1], eigvec[2]);
        let constraint = 4.0 * a1_0 * a1_2 - a1_1 * a1_1;

        if constraint > best_constraint {
            best_constraint = constraint;
            // Recover a2 = -S22⁻¹ S21 a1
            let a1 = DMatrix::from_column_slice(3, 1, &[a1_0, a1_1, a1_2]);
            let a2 = -&s22_inv * &s21 * &a1;
            best_conic = Some([a1_0, a1_1, a1_2, a2[(0, 0)], a2[(1, 0)], a2[(2, 0)]]);
        }
    }

    let conic = best_conic?;
    if best_constraint <= 0.0 {
        return None;
    }

    conic_to_ellipse(conic[0], conic[1], conic[2], conic[3], conic[4], conic[5])
}

/// Find 3 eigenvectors of a 3×3 matrix using QR iteration + SVD null space.
fn find_eigenvectors_3x3(m: &DMatrix<f64>) -> Vec<[f64; 3]> {
    // Use QR algorithm to find eigenvalues
    let mut a = m.clone();
    for _ in 0..50 {
        let qr = a.clone().qr();
        a = qr.r() * qr.q();
    }

    let eigenvalues = [a[(0, 0)], a[(1, 1)], a[(2, 2)]];
    let identity = DMatrix::<f64>::identity(3, 3);

    let mut result = Vec::new();
    for &ev in &eigenvalues {
        // Find null space of (M - λI) via SVD — works for all cases
        let shifted = m - &identity * ev;
        let svd = shifted.svd(true, true);
        if let Some(vt) = &svd.v_t {
            // The last row of Vᵀ corresponds to the smallest singular value
            let last = vt.nrows() - 1;
            result.push([vt[(last, 0)], vt[(last, 1)], vt[(last, 2)]]);
        }
    }
    result
}

/// Convert general conic Ax² + Bxy + Cy² + Dx + Ey + F = 0 to ellipse parameters.
fn conic_to_ellipse(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Option<Ellipse> {
    // Check discriminant: must be an ellipse (4AC - B² > 0)
    let disc = 4.0 * a * c - b * b;
    if disc <= 0.0 {
        return None;
    }

    // Center
    let cx = (b * e - 2.0 * c * d) / disc;
    let cy = (b * d - 2.0 * a * e) / disc;

    // Semi-axes using the matrix form:
    // The eigenvalues of [[a, b/2], [b/2, c]] give the inverse squared semi-axes
    // after accounting for translation.
    let det_m33 = a * (c * f - e * e / 4.0)
        - (b / 2.0) * (b / 2.0 * f - e * d / 4.0)
        + d / 2.0 * (b / 2.0 * e / 2.0 - c * d / 2.0);
    // Use the simpler formulation:
    // semi-axes from eigenvalues of the 2×2 sub-matrix scaled by -det(M33)/det(M22)
    let det_m22 = a * c - (b / 2.0).powi(2);

    if det_m22.abs() < 1e-15 || det_m33.abs() < 1e-15 {
        return None;
    }

    let s1 = a + c;
    let s2 = ((a - c).powi(2) + b * b).sqrt();

    let lambda1 = (s1 + s2) / 2.0;
    let lambda2 = (s1 - s2) / 2.0;

    if lambda1.abs() < 1e-15 || lambda2.abs() < 1e-15 {
        return None;
    }

    let a_axis_sq = -det_m33 / (det_m22 * lambda1);
    let b_axis_sq = -det_m33 / (det_m22 * lambda2);

    if a_axis_sq <= 0.0 || b_axis_sq <= 0.0 {
        return None;
    }

    let semi_a = a_axis_sq.sqrt();
    let semi_b = b_axis_sq.sqrt();

    // Rotation angle
    let angle = if b.abs() < 1e-10 {
        if a < c { 0.0 } else { std::f64::consts::FRAC_PI_2 }
    } else {
        0.5 * (b / (a - c)).atan()
    };

    Some(Ellipse {
        cx,
        cy,
        a: semi_a.max(semi_b),
        b: semi_a.min(semi_b),
        angle,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate points on an ellipse.
    fn ellipse_points(e: &Ellipse, n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / n as f64;
                let x = e.a * t.cos();
                let y = e.b * t.sin();
                let cos = e.angle.cos();
                let sin = e.angle.sin();
                (e.cx + x * cos - y * sin, e.cy + x * sin + y * cos)
            })
            .collect()
    }

    #[test]
    fn fit_circle() {
        let expected = Ellipse { cx: 10.0, cy: 15.0, a: 5.0, b: 5.0, angle: 0.0 };
        let pts = ellipse_points(&expected, 20);
        let result = fit_ellipse(&pts).expect("fit should succeed");
        assert!((result.cx - 10.0).abs() < 0.5, "cx={}", result.cx);
        assert!((result.cy - 15.0).abs() < 0.5, "cy={}", result.cy);
        assert!((result.a - 5.0).abs() < 0.5, "a={}", result.a);
        assert!((result.b - 5.0).abs() < 0.5, "b={}", result.b);
    }

    #[test]
    fn fit_axis_aligned_ellipse() {
        let expected = Ellipse { cx: 20.0, cy: 30.0, a: 12.0, b: 6.0, angle: 0.0 };
        let pts = ellipse_points(&expected, 30);
        let result = fit_ellipse(&pts).expect("fit should succeed");
        assert!((result.cx - 20.0).abs() < 1.0, "cx={}", result.cx);
        assert!((result.cy - 30.0).abs() < 1.0, "cy={}", result.cy);
        assert!((result.a - 12.0).abs() < 1.0, "a={}", result.a);
        assert!((result.b - 6.0).abs() < 1.0, "b={}", result.b);
    }

    #[test]
    fn fit_rotated_ellipse() {
        let expected = Ellipse { cx: 50.0, cy: 50.0, a: 20.0, b: 8.0, angle: PI / 4.0 };
        let pts = ellipse_points(&expected, 40);
        let result = fit_ellipse(&pts).expect("fit should succeed");
        assert!((result.cx - 50.0).abs() < 1.0, "cx={}", result.cx);
        assert!((result.cy - 50.0).abs() < 1.0, "cy={}", result.cy);
        assert!((result.a - 20.0).abs() < 1.5, "a={}", result.a);
        assert!((result.b - 8.0).abs() < 1.5, "b={}", result.b);
    }

    #[test]
    fn fit_noisy_ellipse() {
        let expected = Ellipse { cx: 30.0, cy: 30.0, a: 15.0, b: 10.0, angle: 0.3 };
        let mut pts = ellipse_points(&expected, 50);
        // Add deterministic "noise"
        for (i, p) in pts.iter_mut().enumerate() {
            let noise = ((i * 7 + 3) % 11) as f64 / 11.0 - 0.5; // [-0.5, 0.5]
            p.0 += noise;
            p.1 += noise * 0.7;
        }
        let result = fit_ellipse(&pts).expect("fit should succeed");
        assert!((result.cx - 30.0).abs() < 2.0, "cx={}", result.cx);
        assert!((result.cy - 30.0).abs() < 2.0, "cy={}", result.cy);
    }

    #[test]
    fn too_few_points_returns_none() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(fit_ellipse(&pts).is_none());
    }

    #[test]
    fn contains_point() {
        let e = Ellipse { cx: 0.0, cy: 0.0, a: 10.0, b: 5.0, angle: 0.0 };
        assert!(e.contains(0.0, 0.0));
        assert!(e.contains(5.0, 0.0));
        assert!(!e.contains(11.0, 0.0));
        assert!(e.contains(0.0, 4.0));
        assert!(!e.contains(0.0, 6.0));
    }
}
