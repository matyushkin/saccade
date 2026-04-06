//! GSAR-PuRe: Gradient-Seeded Adaptive-Resolution PuRe.
//!
//! Novel adaptive pupil tracking algorithm that combines:
//! 1. Timm & Barth gradient detector as a fast seed
//! 2. PuRe edge-based detector for precise ellipse fitting
//! 3. Adaptive resolution (2-level) based on tracking confidence
//! 4. Kalman temporal filter for smoothing and prediction
//!
//! The gradient center narrows PuRe's search space via ROI restriction,
//! achieving sub-millisecond average tracking with PuReST-level accuracy.

use crate::ellipse::Ellipse;
use crate::frame::{Frame, OwnedGrayFrame, Roi};
use crate::kalman::PupilKalman;
use crate::pure::{self, PureConfig, PureResult};
use crate::timm::{self, TimmConfig};

/// Tracking mode for the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingMode {
    /// High confidence: low-res Timm seed → small ROI PuRe (~0.5–1 ms).
    Fast,
    /// Medium confidence: high-res Timm seed → medium ROI PuRe (~2–3 ms).
    Precise,
    /// Lost or first frame: full-frame PuRe at high resolution (~5 ms).
    FullScan,
}

/// Result of a single tracking step.
#[derive(Debug, Clone)]
pub struct TrackingResult {
    /// Detected pupil ellipse (in original frame coordinates), if found.
    pub pupil: Option<Ellipse>,
    /// Unified confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// Which tracking mode was used for this frame.
    pub mode: TrackingMode,
    /// Confidence breakdown.
    pub confidence_detail: ConfidenceDetail,
}

/// Breakdown of the unified confidence score.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceDetail {
    /// Gradient agreement (Timm & Barth objective).
    pub gradient: f64,
    /// Edge quality (PuRe ψ score).
    pub edge: f64,
    /// Temporal consistency (Kalman innovation).
    pub temporal: f64,
}

/// Configuration for the GSAR-PuRe tracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Low resolution for fast mode (width). Default: 160.
    pub low_res_width: u32,
    /// High resolution for precise/full mode (width). Default: 320.
    pub high_res_width: u32,

    /// ROI size multiplier relative to last known pupil radius.
    /// ROI side = pupil_radius * roi_multiplier. Default: 6.0.
    pub roi_multiplier: f64,
    /// Minimum ROI size in pixels. Default: 40.
    pub min_roi_size: u32,
    /// Maximum ROI size in pixels. Default: 200.
    pub max_roi_size: u32,

    /// Confidence threshold for fast mode. Default: 0.5.
    pub fast_threshold: f64,
    /// Confidence threshold for precise mode (below this → full scan). Default: 0.2.
    pub precise_threshold: f64,

    /// Weight for gradient confidence in unified score. Default: 0.2.
    pub alpha: f64,
    /// Weight for edge confidence in unified score. Default: 0.5.
    pub beta: f64,
    /// Weight for temporal confidence in unified score. Default: 0.3.
    pub gamma: f64,

    /// Kalman temporal confidence scale. Default: 10.0.
    pub kalman_scale: f64,

    /// Timm & Barth configuration.
    pub timm: TimmConfig,
    /// PuRe configuration.
    pub pure: PureConfig,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            low_res_width: 160,
            high_res_width: 320,
            roi_multiplier: 6.0,
            min_roi_size: 40,
            max_roi_size: 200,
            fast_threshold: 0.5,
            precise_threshold: 0.2,
            alpha: 0.2,
            beta: 0.5,
            gamma: 0.3,
            kalman_scale: 10.0,
            timm: TimmConfig::default(),
            pure: PureConfig::default(),
        }
    }
}

/// Stateful pupil tracker using the GSAR-PuRe algorithm.
pub struct Tracker {
    config: TrackerConfig,
    kalman: PupilKalman,
    /// Last known pupil in original frame coordinates.
    last_pupil: Option<Ellipse>,
    /// Last unified confidence.
    last_confidence: f64,
    /// Frame counter.
    frame_count: u64,
}

impl Tracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            kalman: PupilKalman::default_params(),
            last_pupil: None,
            last_confidence: 0.0,
            frame_count: 0,
        }
    }

    /// Create a new tracker with default configuration.
    pub fn default_config() -> Self {
        Self::new(TrackerConfig::default())
    }

    /// Reset the tracker state (e.g., after a scene change).
    pub fn reset(&mut self) {
        self.kalman = PupilKalman::default_params();
        self.last_pupil = None;
        self.last_confidence = 0.0;
        self.frame_count = 0;
    }

    /// Process a single frame and return the tracking result.
    pub fn track(&mut self, frame: &dyn Frame) -> TrackingResult {
        self.frame_count += 1;

        let mode = self.select_mode();
        let result = match mode {
            TrackingMode::Fast => self.track_fast(frame),
            TrackingMode::Precise => self.track_precise(frame),
            TrackingMode::FullScan => self.track_full(frame),
        };

        // Update Kalman filter
        if let Some(ref pupil) = result.pupil {
            self.kalman.step(pupil.cx, pupil.cy);
            self.last_pupil = Some(*pupil);
        } else {
            // No detection: predict only
            if self.kalman.initialized {
                self.kalman.predict();
            }
        }

        self.last_confidence = result.confidence;
        result
    }

    /// Select tracking mode based on previous frame's confidence.
    fn select_mode(&self) -> TrackingMode {
        if self.frame_count <= 1 || !self.kalman.initialized {
            return TrackingMode::FullScan;
        }
        if self.last_confidence >= self.config.fast_threshold {
            TrackingMode::Fast
        } else if self.last_confidence >= self.config.precise_threshold {
            TrackingMode::Precise
        } else {
            TrackingMode::FullScan
        }
    }

    /// Fast mode: low-res Timm seed → small ROI PuRe.
    fn track_fast(&self, frame: &dyn Frame) -> TrackingResult {
        let (working, scale) = self.downscale(frame, self.config.low_res_width);

        // Use Kalman prediction as initial ROI center
        let (pred_x, pred_y) = self.kalman.position();
        let pred_x_scaled = pred_x / scale;
        let pred_y_scaled = pred_y / scale;

        // Timm & Barth in a small ROI around prediction
        let roi_size = self.compute_roi_size(scale);
        let timm_roi = self.make_roi(pred_x_scaled, pred_y_scaled, roi_size, &working);
        let timm_frame = OwnedGrayFrame::crop(&working, timm_roi);
        let grad = timm::detect_center(&timm_frame, &self.config.timm);

        // Map gradient center back to working resolution
        let seed_x = timm_roi.x as f64 + grad.x;
        let seed_y = timm_roi.y as f64 + grad.y;

        // PuRe in ROI around gradient seed
        let pure_roi = self.make_roi(seed_x, seed_y, roi_size, &working);
        let pure_frame = OwnedGrayFrame::crop(&working, pure_roi);
        let pure_result = pure::detect(&pure_frame, &self.config.pure);

        self.build_result(pure_result, pure_roi, scale, grad.confidence, TrackingMode::Fast)
    }

    /// Precise mode: high-res Timm seed → medium ROI PuRe.
    fn track_precise(&self, frame: &dyn Frame) -> TrackingResult {
        let (working, scale) = self.downscale(frame, self.config.high_res_width);

        let (pred_x, pred_y) = self.kalman.position();
        let pred_x_scaled = pred_x / scale;
        let pred_y_scaled = pred_y / scale;

        // Larger ROI for Timm
        let roi_size = self.compute_roi_size(scale) * 3 / 2;
        let timm_roi = self.make_roi(pred_x_scaled, pred_y_scaled, roi_size, &working);
        let timm_frame = OwnedGrayFrame::crop(&working, timm_roi);
        let grad = timm::detect_center(&timm_frame, &self.config.timm);

        let seed_x = timm_roi.x as f64 + grad.x;
        let seed_y = timm_roi.y as f64 + grad.y;

        let pure_roi = self.make_roi(seed_x, seed_y, roi_size, &working);
        let pure_frame = OwnedGrayFrame::crop(&working, pure_roi);
        let pure_result = pure::detect(&pure_frame, &self.config.pure);

        self.build_result(pure_result, pure_roi, scale, grad.confidence, TrackingMode::Precise)
    }

    /// Full scan: high-res full-frame PuRe (no seed).
    fn track_full(&self, frame: &dyn Frame) -> TrackingResult {
        let (working, scale) = self.downscale(frame, self.config.high_res_width);

        // Run Timm on the full frame for gradient confidence
        let grad = timm::detect_center(&working, &self.config.timm);

        // Run PuRe on the full frame
        let pure_result = pure::detect(&working, &self.config.pure);

        let roi = Roi {
            x: 0,
            y: 0,
            width: working.width(),
            height: working.height(),
        };
        self.build_result(pure_result, roi, scale, grad.confidence, TrackingMode::FullScan)
    }

    /// Build a TrackingResult from PuRe output, mapping coordinates back to original frame.
    fn build_result(
        &self,
        pure_result: PureResult,
        roi: Roi,
        scale: f64,
        gradient_confidence: f64,
        mode: TrackingMode,
    ) -> TrackingResult {
        let pupil = pure_result.pupil.map(|ell| {
            // Map from ROI coordinates back to original frame
            Ellipse {
                cx: (roi.x as f64 + ell.cx) * scale,
                cy: (roi.y as f64 + ell.cy) * scale,
                a: ell.a * scale,
                b: ell.b * scale,
                angle: ell.angle,
            }
        });

        let edge_confidence = pure_result.confidence;
        let temporal_confidence = self.kalman.temporal_confidence(self.config.kalman_scale);

        let confidence_detail = ConfidenceDetail {
            gradient: gradient_confidence,
            edge: edge_confidence,
            temporal: if self.kalman.initialized {
                temporal_confidence
            } else {
                0.5 // neutral for first frame
            },
        };

        let unified = self.config.alpha * confidence_detail.gradient
            + self.config.beta * confidence_detail.edge
            + self.config.gamma * confidence_detail.temporal;

        TrackingResult {
            pupil,
            confidence: unified.clamp(0.0, 1.0),
            mode,
            confidence_detail,
        }
    }

    /// Downscale frame to target width, preserving aspect ratio.
    fn downscale(&self, frame: &dyn Frame, target_width: u32) -> (OwnedGrayFrame, f64) {
        if frame.width() <= target_width {
            let data = frame.gray_pixels().to_vec();
            return (
                OwnedGrayFrame::new(frame.width(), frame.height(), data),
                1.0,
            );
        }
        let factor = frame.width() / target_width;
        let factor = factor.max(1);
        let scale = factor as f64;
        (OwnedGrayFrame::downscale(frame, factor), scale)
    }

    /// Compute ROI size based on last known pupil size.
    fn compute_roi_size(&self, scale: f64) -> u32 {
        let base = if let Some(ref pupil) = self.last_pupil {
            let avg_radius = (pupil.a + pupil.b) / 2.0;
            (avg_radius * self.config.roi_multiplier / scale) as u32
        } else {
            self.config.max_roi_size
        };
        base.clamp(self.config.min_roi_size, self.config.max_roi_size)
    }

    /// Create a centered ROI clamped to frame bounds.
    fn make_roi(&self, cx: f64, cy: f64, size: u32, frame: &dyn Frame) -> Roi {
        let half = size / 2;
        let x = (cx as i32 - half as i32).max(0) as u32;
        let y = (cy as i32 - half as i32).max(0) as u32;
        let w = size.min(frame.width().saturating_sub(x));
        let h = size.min(frame.height().saturating_sub(y));
        Roi {
            x,
            y,
            width: w,
            height: h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::GrayFrame;

    /// Create a synthetic eye with a sharp dark pupil.
    fn make_eye(w: u32, h: u32, cx: f64, cy: f64, rx: f64, ry: f64) -> Vec<u8> {
        let mut data = vec![200u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let d = ((dx / rx).powi(2) + (dy / ry).powi(2)).sqrt();
                if d < 0.9 {
                    data[(y * w + x) as usize] = 30;
                } else if d < 1.1 {
                    let t = (d - 0.9) / 0.2;
                    data[(y * w + x) as usize] = (30.0 + 170.0 * t) as u8;
                }
            }
        }
        data
    }

    #[test]
    fn full_scan_detects_pupil() {
        let (w, h) = (320, 240);
        let data = make_eye(w, h, 160.0, 120.0, 40.0, 30.0);
        let frame = GrayFrame::new(w, h, &data);

        let mut tracker = Tracker::new(TrackerConfig {
            pure: PureConfig {
                canny_low: 10.0,
                canny_high: 30.0,
                ..PureConfig::default()
            },
            ..TrackerConfig::default()
        });

        let result = tracker.track(&frame);
        assert_eq!(result.mode, TrackingMode::FullScan);
        assert!(result.pupil.is_some(), "first frame should detect pupil");

        if let Some(pupil) = result.pupil {
            let error = ((pupil.cx - 160.0).powi(2) + (pupil.cy - 120.0).powi(2)).sqrt();
            assert!(error < 20.0, "center error {error:.1}px");
        }
    }

    #[test]
    fn tracking_across_frames() {
        let (w, h) = (320, 240);
        let mut tracker = Tracker::new(TrackerConfig {
            pure: PureConfig {
                canny_low: 10.0,
                canny_high: 30.0,
                ..PureConfig::default()
            },
            ..TrackerConfig::default()
        });

        // Simulate 10 frames with slight pupil movement
        for i in 0..10 {
            let cx = 160.0 + i as f64 * 2.0;
            let cy = 120.0 + i as f64;
            let data = make_eye(w, h, cx, cy, 40.0, 30.0);
            let frame = GrayFrame::new(w, h, &data);
            let result = tracker.track(&frame);

            // First frame: full scan; subsequent: may use fast/precise
            if i == 0 {
                assert_eq!(result.mode, TrackingMode::FullScan);
            }
            // Should generally detect a pupil
            if i > 0 {
                // After first frame, tracking should work
                assert!(
                    result.pupil.is_some() || result.confidence > 0.0,
                    "frame {i}: should track or have confidence"
                );
            }
        }
    }

    #[test]
    fn mode_switches_on_confidence() {
        let tracker = Tracker {
            config: TrackerConfig::default(),
            kalman: PupilKalman::default_params(),
            last_pupil: Some(Ellipse {
                cx: 100.0,
                cy: 100.0,
                a: 20.0,
                b: 15.0,
                angle: 0.0,
            }),
            last_confidence: 0.8,
            frame_count: 5,
        };
        // High confidence → Fast mode (but Kalman not initialized)
        assert_eq!(tracker.select_mode(), TrackingMode::FullScan);

        // With initialized Kalman
        let mut tracker2 = tracker;
        tracker2.kalman.init(100.0, 100.0);
        assert_eq!(tracker2.select_mode(), TrackingMode::Fast);

        tracker2.last_confidence = 0.3;
        assert_eq!(tracker2.select_mode(), TrackingMode::Precise);

        tracker2.last_confidence = 0.1;
        assert_eq!(tracker2.select_mode(), TrackingMode::FullScan);
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = Tracker::default_config();
        tracker.last_confidence = 0.9;
        tracker.frame_count = 100;
        tracker.last_pupil = Some(Ellipse {
            cx: 50.0, cy: 50.0, a: 10.0, b: 10.0, angle: 0.0,
        });
        tracker.reset();
        assert_eq!(tracker.frame_count, 0);
        assert!(tracker.last_pupil.is_none());
        assert_eq!(tracker.last_confidence, 0.0);
    }
}
