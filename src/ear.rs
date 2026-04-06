//! Eye Aspect Ratio (EAR) computation from facial landmarks.
//!
//! EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
//!
//! Where p1-p6 are the 6 eye contour landmarks (iBUG 68-point scheme):
//! - p1 (outer corner), p2 (upper-outer), p3 (upper-inner)
//! - p4 (inner corner), p5 (lower-inner), p6 (lower-outer)
//!
//! Open eye: EAR ≈ 0.25-0.35
//! Closed eye: EAR ≈ 0.05-0.15

/// Compute Eye Aspect Ratio from 6 landmark points.
///
/// Points should be in order: [outer_corner, upper_outer, upper_inner,
/// inner_corner, lower_inner, lower_outer].
///
/// iBUG 68-point: right eye = landmarks[36..42], left eye = landmarks[42..48].
pub fn compute_ear(points: &[(f32, f32); 6]) -> f32 {
    let p1 = points[0]; // outer corner
    let p2 = points[1]; // upper outer
    let p3 = points[2]; // upper inner
    let p4 = points[3]; // inner corner
    let p5 = points[4]; // lower inner
    let p6 = points[5]; // lower outer

    let vertical_1 = dist(p2, p6);
    let vertical_2 = dist(p3, p5);
    let horizontal = dist(p1, p4);

    if horizontal < 1e-6 {
        return 0.0;
    }

    (vertical_1 + vertical_2) / (2.0 * horizontal)
}

/// Compute EAR for both eyes from 68 iBUG landmarks.
///
/// Returns (right_ear, left_ear).
pub fn compute_ear_from_landmarks(landmarks: &[(f32, f32)]) -> Option<(f32, f32)> {
    if landmarks.len() < 48 {
        return None;
    }

    let right_eye: [(f32, f32); 6] = [
        landmarks[36], landmarks[37], landmarks[38],
        landmarks[39], landmarks[40], landmarks[41],
    ];
    let left_eye: [(f32, f32); 6] = [
        landmarks[42], landmarks[43], landmarks[44],
        landmarks[45], landmarks[46], landmarks[47],
    ];

    Some((compute_ear(&right_eye), compute_ear(&left_eye)))
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_eye_has_high_ear() {
        // Simulate open eye: vertical gaps ~= horizontal/3
        let points: [(f32, f32); 6] = [
            (0.0, 0.0),   // p1: outer corner
            (1.0, -1.5),  // p2: upper outer
            (2.0, -1.5),  // p3: upper inner
            (3.0, 0.0),   // p4: inner corner
            (2.0, 1.5),   // p5: lower inner
            (1.0, 1.5),   // p6: lower outer
        ];
        let ear = compute_ear(&points);
        assert!(ear > 0.8, "open eye EAR={ear}");
    }

    #[test]
    fn closed_eye_has_low_ear() {
        // Simulate closed eye: vertical gaps ~= 0
        let points: [(f32, f32); 6] = [
            (0.0, 0.0),
            (1.0, -0.1),
            (2.0, -0.1),
            (3.0, 0.0),
            (2.0, 0.1),
            (1.0, 0.1),
        ];
        let ear = compute_ear(&points);
        assert!(ear < 0.1, "closed eye EAR={ear}");
    }
}
