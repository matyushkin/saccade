//! # Saccade
//!
//! Pure Rust eye tracking library.
//!
//! ## Features
//!
//! - Pupil detection (gradient-based and edge-based)
//! - Gaze estimation
//! - Fixation and saccade classification
//! - Blink detection
//! - Kalman temporal tracking

pub mod frame;
pub mod detect;
pub mod gaze;
pub mod blink;
pub mod classify;
pub mod kalman;
pub mod timm;
pub mod ellipse;
pub mod edge;
pub mod pure;
pub mod tracker;
pub mod ear;
pub mod calibration;
pub mod preprocess;
pub mod one_euro;
pub mod ridge;
pub mod calib_state;
pub mod session;
