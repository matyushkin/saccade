//! Kalman filter for temporal pupil tracking.
//!
//! Tracks pupil ellipse state (center, semi-axes, velocity) across frames
//! and provides prediction, smoothing, and anomaly detection.

use nalgebra::{Matrix4, Matrix4x2, Matrix2x4, Matrix2, Vector4, Vector2};

/// Kalman filter state for tracking a single pupil.
///
/// State vector: `[cx, cy, vx, vy]`
/// - `(cx, cy)`: pupil center position
/// - `(vx, vy)`: velocity (pixels per frame)
///
/// Measurement vector: `[cx, cy]`
#[derive(Debug, Clone)]
pub struct PupilKalman {
    /// State estimate: [cx, cy, vx, vy].
    pub state: Vector4<f64>,
    /// Error covariance matrix (4×4).
    pub covariance: Matrix4<f64>,
    /// State transition matrix.
    transition: Matrix4<f64>,
    /// Measurement matrix (2×4): extracts position from state.
    measurement: Matrix2x4<f64>,
    /// Process noise covariance.
    process_noise: Matrix4<f64>,
    /// Measurement noise covariance.
    measurement_noise: Matrix2<f64>,
    /// Last innovation (measurement residual). Used for confidence estimation.
    pub innovation: Vector2<f64>,
    /// Whether the filter has been initialized with a measurement.
    pub initialized: bool,
}

impl PupilKalman {
    /// Create a new Kalman filter with given noise parameters.
    ///
    /// - `process_noise_pos`: position process noise variance (higher = more responsive)
    /// - `process_noise_vel`: velocity process noise variance
    /// - `measurement_noise`: measurement noise variance (higher = more smoothing)
    pub fn new(process_noise_pos: f64, process_noise_vel: f64, measurement_noise: f64) -> Self {
        // State transition: constant velocity model
        // [cx]   [1 0 1 0] [cx]
        // [cy] = [0 1 0 1] [cy]
        // [vx]   [0 0 1 0] [vx]
        // [vy]   [0 0 0 1] [vy]
        let transition = Matrix4::new(
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // Measurement matrix: we observe position only
        let measurement = Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );

        let process_noise = Matrix4::new(
            process_noise_pos, 0.0, 0.0, 0.0,
            0.0, process_noise_pos, 0.0, 0.0,
            0.0, 0.0, process_noise_vel, 0.0,
            0.0, 0.0, 0.0, process_noise_vel,
        );

        let measurement_noise = Matrix2::new(
            measurement_noise, 0.0,
            0.0, measurement_noise,
        );

        Self {
            state: Vector4::zeros(),
            covariance: Matrix4::identity() * 1000.0, // high initial uncertainty
            transition,
            measurement,
            process_noise,
            measurement_noise,
            innovation: Vector2::zeros(),
            initialized: false,
        }
    }

    /// Create with sensible defaults for pupil tracking.
    pub fn default_params() -> Self {
        Self::new(1.0, 0.5, 2.0)
    }

    /// Initialize the filter with the first measurement.
    pub fn init(&mut self, cx: f64, cy: f64) {
        self.state = Vector4::new(cx, cy, 0.0, 0.0);
        self.covariance = Matrix4::identity() * 10.0;
        self.innovation = Vector2::zeros();
        self.initialized = true;
    }

    /// Predict the next state (call before update).
    pub fn predict(&mut self) {
        // x' = F * x
        self.state = self.transition * self.state;
        // P' = F * P * Fᵀ + Q
        self.covariance =
            self.transition * self.covariance * self.transition.transpose() + self.process_noise;
    }

    /// Update the state with a new measurement.
    pub fn update(&mut self, cx: f64, cy: f64) {
        if !self.initialized {
            self.init(cx, cy);
            return;
        }

        let z = Vector2::new(cx, cy);

        // Innovation: y = z - H * x
        self.innovation = z - self.measurement * self.state;

        // Innovation covariance: S = H * P * Hᵀ + R
        let s: Matrix2<f64> =
            self.measurement * self.covariance * self.measurement.transpose()
                + self.measurement_noise;

        // Kalman gain: K = P * Hᵀ * S⁻¹
        let s_inv = s.try_inverse().unwrap_or(Matrix2::identity());
        let k: Matrix4x2<f64> =
            self.covariance * self.measurement.transpose() * s_inv;

        // Update state: x = x + K * y
        self.state += k * self.innovation;

        // Update covariance: P = (I - K * H) * P
        let i = Matrix4::identity();
        self.covariance = (i - k * self.measurement) * self.covariance;
    }

    /// Predict and update in one step.
    pub fn step(&mut self, cx: f64, cy: f64) {
        if !self.initialized {
            self.init(cx, cy);
            return;
        }
        self.predict();
        self.update(cx, cy);
    }

    /// Get the current predicted position.
    pub fn position(&self) -> (f64, f64) {
        (self.state[0], self.state[1])
    }

    /// Get the current predicted velocity.
    pub fn velocity(&self) -> (f64, f64) {
        (self.state[2], self.state[3])
    }

    /// Temporal confidence based on innovation magnitude.
    /// Returns a value in [0.0, 1.0] where 1.0 = highly consistent tracking.
    ///
    /// `scale` controls sensitivity: larger scale = more forgiving of large innovations.
    pub fn temporal_confidence(&self, scale: f64) -> f64 {
        let inn_mag = (self.innovation[0].powi(2) + self.innovation[1].powi(2)).sqrt();
        (-inn_mag / scale).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_target_converges() {
        let mut kf = PupilKalman::default_params();
        // Feed the same position repeatedly
        for _ in 0..20 {
            kf.step(100.0, 50.0);
        }
        let (cx, cy) = kf.position();
        assert!((cx - 100.0).abs() < 0.5, "cx={cx}");
        assert!((cy - 50.0).abs() < 0.5, "cy={cy}");
        assert!(kf.temporal_confidence(5.0) > 0.9);
    }

    #[test]
    fn constant_velocity_tracking() {
        let mut kf = PupilKalman::default_params();
        // Target moves at 2px/frame in X
        for i in 0..30 {
            let x = 100.0 + 2.0 * i as f64;
            kf.step(x, 50.0);
        }
        let (vx, vy) = kf.velocity();
        assert!((vx - 2.0).abs() < 0.5, "vx={vx}");
        assert!(vy.abs() < 0.5, "vy={vy}");
    }

    #[test]
    fn jump_detected_as_low_confidence() {
        let mut kf = PupilKalman::default_params();
        for _ in 0..20 {
            kf.step(100.0, 50.0);
        }
        let conf_before = kf.temporal_confidence(5.0);
        // Sudden jump
        kf.step(200.0, 150.0);
        let conf_after = kf.temporal_confidence(5.0);
        assert!(conf_after < conf_before, "confidence should drop on jump");
        assert!(conf_after < 0.5, "conf_after={conf_after}");
    }

    #[test]
    fn prediction_without_update() {
        let mut kf = PupilKalman::default_params();
        for i in 0..20 {
            kf.step(100.0 + 2.0 * i as f64, 50.0);
        }
        // Predict without measurement (simulating lost detection)
        kf.predict();
        let (cx, _) = kf.position();
        // Should extrapolate forward
        assert!(cx > 138.0, "cx={cx} should extrapolate");
    }
}
