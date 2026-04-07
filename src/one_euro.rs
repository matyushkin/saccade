//! 1€ filter for smoothing noisy signals with adaptive low-pass cutoff.
//!
//! Casiez, Roussel, Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter
//! for Noisy Input in Interactive Systems", CHI 2012.
//!
//! The cutoff frequency adapts to the signal's speed:
//! - Low speed → low cutoff → more smoothing (reduces jitter)
//! - High speed → high cutoff → less smoothing (preserves responsiveness)

/// Single-axis 1€ filter.
pub struct OneEuroFilter {
    /// Minimum cutoff frequency (Hz). Lower = more smoothing at rest.
    pub min_cutoff: f64,
    /// Speed coefficient. Higher = faster response to fast motion.
    pub beta: f64,
    /// Derivative cutoff frequency (Hz).
    pub d_cutoff: f64,

    x_prev: Option<f64>,
    dx_prev: f64,
    t_prev: Option<f64>,
}

impl OneEuroFilter {
    /// Create a new 1€ filter.
    ///
    /// Typical values:
    /// - `min_cutoff = 1.0` Hz (smooths jitter at rest)
    /// - `beta = 0.007` (typical, increase for more responsiveness)
    /// - `d_cutoff = 1.0` Hz (derivative smoothing)
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
        self.t_prev = None;
    }

    /// Filter a new sample at given time (seconds).
    pub fn filter(&mut self, x: f64, t: f64) -> f64 {
        let dt = match self.t_prev {
            Some(prev_t) => (t - prev_t).max(1e-6),
            None => {
                self.x_prev = Some(x);
                self.t_prev = Some(t);
                return x;
            }
        };
        let x_prev = self.x_prev.unwrap_or(x);

        // Estimate derivative
        let dx = (x - x_prev) / dt;
        let a_d = smoothing_factor(dt, self.d_cutoff);
        let dx_smooth = a_d * dx + (1.0 - a_d) * self.dx_prev;

        // Adaptive cutoff based on speed
        let cutoff = self.min_cutoff + self.beta * dx_smooth.abs();
        let a = smoothing_factor(dt, cutoff);
        let x_smooth = a * x + (1.0 - a) * x_prev;

        self.x_prev = Some(x_smooth);
        self.dx_prev = dx_smooth;
        self.t_prev = Some(t);
        x_smooth
    }
}

fn smoothing_factor(dt: f64, cutoff: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

/// 2D 1€ filter (independent filters for X and Y).
pub struct OneEuroFilter2D {
    pub x: OneEuroFilter,
    pub y: OneEuroFilter,
}

impl OneEuroFilter2D {
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            x: OneEuroFilter::new(min_cutoff, beta, d_cutoff),
            y: OneEuroFilter::new(min_cutoff, beta, d_cutoff),
        }
    }

    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
    }

    /// Filter a 2D point at time `t` (seconds).
    pub fn filter(&mut self, point: (f64, f64), t: f64) -> (f64, f64) {
        (self.x.filter(point.0, t), self.y.filter(point.1, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_constant_signal() {
        let mut f = OneEuroFilter::new(1.0, 0.007, 1.0);
        let mut last = 0.0;
        for i in 0..30 {
            last = f.filter(100.0, i as f64 * 0.033);
        }
        assert!((last - 100.0).abs() < 0.5, "last={last}");
    }

    #[test]
    fn smooths_noise_at_rest() {
        let mut f = OneEuroFilter::new(1.0, 0.007, 1.0);
        // Feed noisy samples around 50.0
        let samples = [50.0, 51.5, 48.5, 52.0, 49.0, 51.0, 50.5];
        let mut last = 0.0;
        for (i, &s) in samples.iter().enumerate() {
            last = f.filter(s, i as f64 * 0.033);
        }
        // Heavy smoothing should bring this close to ~50
        assert!((last - 50.0).abs() < 1.5, "last={last}");
    }

    #[test]
    fn responds_to_fast_motion() {
        let mut f = OneEuroFilter::new(1.0, 0.1, 1.0);
        // Start at 0, jump to 100 rapidly
        for i in 0..5 {
            f.filter(0.0, i as f64 * 0.033);
        }
        // Fast ramp from 0 to 100
        let mut last = 0.0;
        for i in 0..10 {
            let x = i as f64 * 10.0;
            last = f.filter(x, (5 + i) as f64 * 0.033);
        }
        // Should follow the ramp within reason
        assert!(last > 60.0, "last={last} should follow fast motion");
    }

    #[test]
    fn filter_2d_works() {
        let mut f = OneEuroFilter2D::new(1.0, 0.007, 1.0);
        let p = f.filter((100.0, 50.0), 0.0);
        assert_eq!(p, (100.0, 50.0));
        let p = f.filter((100.0, 50.0), 0.033);
        assert!((p.0 - 100.0).abs() < 1.0);
        assert!((p.1 - 50.0).abs() < 1.0);
    }
}
