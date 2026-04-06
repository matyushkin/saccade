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
