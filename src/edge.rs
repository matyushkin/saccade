//! Canny edge detection (pure Rust implementation).

use crate::frame::Frame;

/// Run Canny edge detection on a grayscale frame.
///
/// Returns a binary edge map (255 = edge, 0 = not edge) as a Vec<u8>.
///
/// - `low_threshold`: hysteresis low threshold (e.g., 20.0)
/// - `high_threshold`: hysteresis high threshold (e.g., 50.0)
pub fn canny(frame: &dyn Frame, low_threshold: f32, high_threshold: f32) -> Vec<u8> {
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let pixels = frame.gray_pixels();

    // Step 1: Gaussian blur (5×5 approximation via two box blurs)
    let blurred = gaussian_blur_5x5(pixels, w, h);

    // Step 2: Sobel gradients
    let (gx, gy, magnitude, direction) = sobel_gradients(&blurred, w, h);
    let _ = (gx, gy); // used implicitly via magnitude/direction

    // Step 3: Non-maximum suppression
    let nms = non_max_suppression(&magnitude, &direction, w, h);

    // Step 4: Double threshold + hysteresis
    hysteresis(&nms, w, h, low_threshold, high_threshold)
}

/// Extract connected edge segments from a binary edge map.
///
/// Returns a list of segments, each being a list of (x, y) coordinates.
/// Segments shorter than `min_length` are discarded.
pub fn extract_edge_segments(edge_map: &[u8], w: usize, h: usize, min_length: usize) -> Vec<Vec<(u32, u32)>> {
    let mut visited = vec![false; w * h];
    let mut segments = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if edge_map[idx] == 255 && !visited[idx] {
                // BFS/flood-fill to trace connected edge pixels
                let mut segment = Vec::new();
                let mut stack = vec![(x as u32, y as u32)];
                while let Some((px, py)) = stack.pop() {
                    let pidx = py as usize * w + px as usize;
                    if visited[pidx] {
                        continue;
                    }
                    visited[pidx] = true;
                    segment.push((px, py));

                    // 8-connected neighbors
                    for dy in [-1i32, 0, 1] {
                        for dx in [-1i32, 0, 1] {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = px as i32 + dx;
                            let ny = py as i32 + dy;
                            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                                let nidx = ny as usize * w + nx as usize;
                                if edge_map[nidx] == 255 && !visited[nidx] {
                                    stack.push((nx as u32, ny as u32));
                                }
                            }
                        }
                    }
                }
                if segment.len() >= min_length {
                    segments.push(segment);
                }
            }
        }
    }

    segments
}

fn gaussian_blur_5x5(pixels: &[u8], w: usize, h: usize) -> Vec<f32> {
    // Approximation: two passes of 3×3 box blur
    let mut buf: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    let mut tmp = vec![0.0f32; w * h];

    for _pass in 0..2 {
        // Horizontal
        for y in 0..h {
            for x in 0..w {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 2).min(w);
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for bx in x0..x1 {
                    sum += buf[y * w + bx];
                    count += 1;
                }
                tmp[y * w + x] = sum / count as f32;
            }
        }
        // Vertical
        for y in 0..h {
            for x in 0..w {
                let y0 = y.saturating_sub(1);
                let y1 = (y + 2).min(h);
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for by in y0..y1 {
                    sum += tmp[by * w + x];
                    count += 1;
                }
                buf[y * w + x] = sum / count as f32;
            }
        }
    }
    buf
}

fn sobel_gradients(img: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    let mut magnitude = vec![0.0f32; w * h];
    let mut direction = vec![0.0f32; w * h];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let dx = -img[(y - 1) * w + (x - 1)]
                + img[(y - 1) * w + (x + 1)]
                - 2.0 * img[y * w + (x - 1)]
                + 2.0 * img[y * w + (x + 1)]
                - img[(y + 1) * w + (x - 1)]
                + img[(y + 1) * w + (x + 1)];

            let dy = -img[(y - 1) * w + (x - 1)]
                - 2.0 * img[(y - 1) * w + x]
                - img[(y - 1) * w + (x + 1)]
                + img[(y + 1) * w + (x - 1)]
                + 2.0 * img[(y + 1) * w + x]
                + img[(y + 1) * w + (x + 1)];

            gx[idx] = dx;
            gy[idx] = dy;
            magnitude[idx] = (dx * dx + dy * dy).sqrt();
            direction[idx] = dy.atan2(dx);
        }
    }

    (gx, gy, magnitude, direction)
}

fn non_max_suppression(magnitude: &[f32], direction: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; w * h];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let mag = magnitude[idx];
            if mag < 1e-5 {
                continue;
            }

            // Quantize direction to 4 main angles (0°, 45°, 90°, 135°)
            let angle = direction[idx].to_degrees().rem_euclid(180.0);
            let (n1, n2) = if angle < 22.5 || angle >= 157.5 {
                // Horizontal edge → compare up/down
                (magnitude[(y - 1) * w + x], magnitude[(y + 1) * w + x])
            } else if angle < 67.5 {
                // 45° edge
                (magnitude[(y - 1) * w + (x + 1)], magnitude[(y + 1) * w + (x - 1)])
            } else if angle < 112.5 {
                // Vertical edge → compare left/right
                (magnitude[y * w + (x - 1)], magnitude[y * w + (x + 1)])
            } else {
                // 135° edge
                (magnitude[(y - 1) * w + (x - 1)], magnitude[(y + 1) * w + (x + 1)])
            };

            if mag >= n1 && mag >= n2 {
                result[idx] = mag;
            }
        }
    }

    result
}

fn hysteresis(nms: &[f32], w: usize, h: usize, low: f32, high: f32) -> Vec<u8> {
    let mut output = vec![0u8; w * h];

    // Mark strong and weak edges
    const STRONG: u8 = 255;
    const WEAK: u8 = 128;

    for i in 0..w * h {
        if nms[i] >= high {
            output[i] = STRONG;
        } else if nms[i] >= low {
            output[i] = WEAK;
        }
    }

    // Promote weak edges connected to strong edges
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                if output[idx] != WEAK {
                    continue;
                }
                // Check 8 neighbors for strong edge
                let has_strong = (-1i32..=1).any(|dy| {
                    (-1i32..=1).any(|dx| {
                        if dx == 0 && dy == 0 {
                            return false;
                        }
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        output[ny * w + nx] == STRONG
                    })
                });
                if has_strong {
                    output[idx] = STRONG;
                    changed = true;
                }
            }
        }
    }

    // Remove remaining weak edges
    for p in &mut output {
        if *p != STRONG {
            *p = 0;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::GrayFrame;

    #[test]
    fn canny_detects_edges_on_step_image() {
        // Left half dark, right half bright
        let w = 40u32;
        let h = 20u32;
        let data: Vec<u8> = (0..h)
            .flat_map(|_| (0..w).map(|x| if x < w / 2 { 30 } else { 220 }))
            .collect();
        let frame = GrayFrame::new(w, h, &data);
        let edges = canny(&frame, 20.0, 50.0);

        // Should have edges near the middle column (x ≈ 20)
        let edge_count: usize = edges.iter().filter(|&&p| p == 255).count();
        assert!(edge_count > 0, "should detect at least some edges");

        // Most edges should be near x=20
        let center_edges: usize = (0..h as usize)
            .filter(|&y| {
                (18..23).any(|x| edges[y * w as usize + x] == 255)
            })
            .count();
        assert!(
            center_edges > h as usize / 2,
            "most edges should be near the center: {center_edges}/{h}"
        );
    }

    #[test]
    fn extract_segments_from_edge_map() {
        let w = 10usize;
        let h = 10usize;
        let mut edge_map = vec![0u8; w * h];
        // Draw a small connected segment
        for x in 3..7 {
            edge_map[5 * w + x] = 255;
        }
        let segments = extract_edge_segments(&edge_map, w, h, 2);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].len(), 4);
    }

    #[test]
    fn short_segments_filtered() {
        let w = 10usize;
        let h = 10usize;
        let mut edge_map = vec![0u8; w * h];
        // Single pixel "segment"
        edge_map[5 * w + 5] = 255;
        let segments = extract_edge_segments(&edge_map, w, h, 3);
        assert!(segments.is_empty());
    }
}
