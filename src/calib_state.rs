//! Calibration state machine — separated from UI so it can be unit tested.

/// Phase of the calibration/tracking session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Calibrating,
    /// After calibration: user looks at center for N seconds, we measure error.
    Validating,
    Running,
}

/// Event that can be sent to the calibration state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Attempt to capture a sample (e.g., user pressed space).
    /// Returns true if captured, false if rejected (no features, too fast).
    CaptureSample,
    /// Restart calibration from scratch.
    Restart,
}

/// Result of processing an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventResult {
    /// Sample captured at point (target_index, sample_index_within_point).
    SampleCaptured { point: usize, sample: u32 },
    /// Advanced to the next calibration point.
    NextPoint { point: usize },
    /// Calibration complete — now in Running phase.
    CalibrationComplete,
    /// Restart — reset to calibrating from point 0.
    Restarted,
    /// Event was rejected.
    Rejected,
}

/// Pure state machine for calibration logic.
///
/// Does NOT depend on timing, clicks, or any UI — just models the sequence of
/// "capture sample" events flowing through a fixed number of points.
pub struct CalibrationState {
    phase: Phase,
    num_points: usize,
    samples_per_point: u32,
    current_point: usize,
    samples_at_current: u32,
    total_samples: u64,
}

impl CalibrationState {
    pub fn new(num_points: usize, samples_per_point: u32) -> Self {
        Self {
            phase: Phase::Idle,
            num_points,
            samples_per_point,
            current_point: 0,
            samples_at_current: 0,
            total_samples: 0,
        }
    }

    pub fn phase(&self) -> Phase { self.phase }
    pub fn current_point(&self) -> usize { self.current_point }
    pub fn samples_at_current(&self) -> u32 { self.samples_at_current }
    pub fn total_samples(&self) -> u64 { self.total_samples }

    /// Start calibration (transition from Idle to Calibrating).
    pub fn start(&mut self) {
        self.phase = Phase::Calibrating;
        self.current_point = 0;
        self.samples_at_current = 0;
    }

    /// Process a capture event. `has_features` indicates whether we have valid
    /// eye features to capture at this moment.
    pub fn handle_capture(&mut self, has_features: bool) -> EventResult {
        if self.phase != Phase::Calibrating {
            return EventResult::Rejected;
        }
        if !has_features {
            return EventResult::Rejected;
        }
        if self.current_point >= self.num_points {
            return EventResult::Rejected;
        }

        self.samples_at_current += 1;
        self.total_samples += 1;

        if self.samples_at_current >= self.samples_per_point {
            // Move to next point
            self.current_point += 1;
            self.samples_at_current = 0;
            if self.current_point >= self.num_points {
                // After last point, enter Validating (not Running directly)
                self.phase = Phase::Validating;
                return EventResult::CalibrationComplete;
            } else {
                return EventResult::NextPoint { point: self.current_point };
            }
        }

        EventResult::SampleCaptured {
            point: self.current_point,
            sample: self.samples_at_current,
        }
    }

    /// Transition from Validating to Running.
    pub fn finish_validation(&mut self) {
        if self.phase == Phase::Validating {
            self.phase = Phase::Running;
        }
    }

    pub fn restart(&mut self) -> EventResult {
        self.phase = Phase::Calibrating;
        self.current_point = 0;
        self.samples_at_current = 0;
        self.total_samples = 0;
        EventResult::Restarted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_in_idle_phase() {
        let s = CalibrationState::new(9, 5);
        assert_eq!(s.phase(), Phase::Idle);
        assert_eq!(s.current_point(), 0);
        assert_eq!(s.samples_at_current(), 0);
    }

    #[test]
    fn capture_rejected_when_idle() {
        let mut s = CalibrationState::new(9, 5);
        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::Rejected);
    }

    #[test]
    fn start_transitions_to_calibrating() {
        let mut s = CalibrationState::new(9, 5);
        s.start();
        assert_eq!(s.phase(), Phase::Calibrating);
    }

    #[test]
    fn capture_without_features_rejected() {
        let mut s = CalibrationState::new(9, 5);
        s.start();
        let r = s.handle_capture(false);
        assert_eq!(r, EventResult::Rejected);
        assert_eq!(s.samples_at_current(), 0);
    }

    #[test]
    fn captures_advance_within_point() {
        let mut s = CalibrationState::new(9, 5);
        s.start();

        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::SampleCaptured { point: 0, sample: 1 });
        assert_eq!(s.samples_at_current(), 1);

        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::SampleCaptured { point: 0, sample: 2 });

        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::SampleCaptured { point: 0, sample: 3 });

        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::SampleCaptured { point: 0, sample: 4 });
    }

    #[test]
    fn fifth_capture_advances_to_next_point() {
        let mut s = CalibrationState::new(9, 5);
        s.start();
        for _ in 0..4 { s.handle_capture(true); }
        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::NextPoint { point: 1 });
        assert_eq!(s.current_point(), 1);
        assert_eq!(s.samples_at_current(), 0);
    }

    #[test]
    fn full_calibration_flow() {
        let mut s = CalibrationState::new(3, 2);
        s.start();

        // Point 0
        assert_eq!(s.handle_capture(true), EventResult::SampleCaptured { point: 0, sample: 1 });
        assert_eq!(s.handle_capture(true), EventResult::NextPoint { point: 1 });

        // Point 1
        assert_eq!(s.handle_capture(true), EventResult::SampleCaptured { point: 1, sample: 1 });
        assert_eq!(s.handle_capture(true), EventResult::NextPoint { point: 2 });

        // Point 2 — last one
        assert_eq!(s.handle_capture(true), EventResult::SampleCaptured { point: 2, sample: 1 });
        assert_eq!(s.handle_capture(true), EventResult::CalibrationComplete);

        assert_eq!(s.phase(), Phase::Validating);
        assert_eq!(s.total_samples(), 6);

        s.finish_validation();
        assert_eq!(s.phase(), Phase::Running);
    }

    #[test]
    fn capture_rejected_after_complete() {
        let mut s = CalibrationState::new(2, 2);
        s.start();
        for _ in 0..4 { s.handle_capture(true); }
        assert_eq!(s.phase(), Phase::Validating);
        let r = s.handle_capture(true);
        assert_eq!(r, EventResult::Rejected);
    }

    #[test]
    fn restart_resets_state() {
        let mut s = CalibrationState::new(9, 5);
        s.start();
        s.handle_capture(true);
        s.handle_capture(true);
        assert_eq!(s.samples_at_current(), 2);

        s.restart();
        assert_eq!(s.phase(), Phase::Calibrating);
        assert_eq!(s.current_point(), 0);
        assert_eq!(s.samples_at_current(), 0);
        assert_eq!(s.total_samples(), 0);
    }

    #[test]
    fn rejected_captures_dont_advance() {
        let mut s = CalibrationState::new(9, 5);
        s.start();
        // Rapid sequence with mixed feature availability
        s.handle_capture(false); // rejected
        s.handle_capture(true);  // captured
        s.handle_capture(false); // rejected
        s.handle_capture(true);  // captured
        assert_eq!(s.samples_at_current(), 2);
        assert_eq!(s.total_samples(), 2);
    }

    #[test]
    fn zero_samples_per_point_edge_case() {
        // Degenerate config — any capture immediately advances
        let mut s = CalibrationState::new(2, 1);
        s.start();
        assert_eq!(s.handle_capture(true), EventResult::NextPoint { point: 1 });
        assert_eq!(s.handle_capture(true), EventResult::CalibrationComplete);
    }

    #[test]
    fn validation_finishes_to_running() {
        let mut s = CalibrationState::new(1, 1);
        s.start();
        s.handle_capture(true);
        assert_eq!(s.phase(), Phase::Validating);
        s.finish_validation();
        assert_eq!(s.phase(), Phase::Running);
    }
}
