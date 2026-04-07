//! Image preprocessing for eye region enhancement.
//!
//! Provides:
//! - Histogram equalization with clip limit (CLAHE-style)
//! - Glint removal (bright reflection masking)
//! - Bilinear upscaling
//! - Gaussian blur

/// Clipped histogram equalization — better than min-max stretch for uneven lighting.
///
/// Works like CLAHE but globally (single tile) — suitable for small eye ROIs.
/// `clip_limit` controls how much contrast to boost (typical: 0.01-0.05).
pub fn clahe_global(pixels: &mut [u8], clip_limit: f64) {
    if pixels.is_empty() {
        return;
    }

    let n = pixels.len();
    let mut hist = [0u32; 256];
    for &p in pixels.iter() {
        hist[p as usize] += 1;
    }

    // Clip histogram to limit
    let limit = (clip_limit * n as f64) as u32;
    let mut excess = 0u32;
    for bin in hist.iter_mut() {
        if *bin > limit {
            excess += *bin - limit;
            *bin = limit;
        }
    }

    // Redistribute excess uniformly
    let redist = excess / 256;
    for bin in hist.iter_mut() {
        *bin += redist;
    }

    // Build cumulative distribution function (CDF)
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
    let cdf_max = cdf[255];
    if cdf_max <= cdf_min {
        return;
    }
    let range = (cdf_max - cdf_min) as f64;

    // Apply mapping
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = (cdf[i].saturating_sub(cdf_min)) as f64 / range;
        lut[i] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    for p in pixels.iter_mut() {
        *p = lut[*p as usize];
    }
}

/// Remove glints (bright reflections) by replacing bright pixels with local median.
///
/// Pixels brighter than `mean + threshold_std * std` are considered glints.
/// Typical threshold_std: 2.0-3.0.
pub fn remove_glints(pixels: &mut [u8], width: usize, height: usize, threshold_std: f64) {
    if pixels.len() != width * height || pixels.is_empty() {
        return;
    }

    let n = pixels.len() as f64;
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = pixels.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / n;
    let std_dev = var.sqrt();
    let threshold = (mean + threshold_std * std_dev).min(255.0) as u8;

    // Build a mask of glint pixels and replace them with local median (3x3 window)
    let mut glint_mask = vec![false; width * height];
    let mut glint_count = 0;
    for (i, &p) in pixels.iter().enumerate() {
        if p >= threshold && p > 180 {
            glint_mask[i] = true;
            glint_count += 1;
        }
    }

    if glint_count == 0 {
        return;
    }

    let original = pixels.to_vec();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !glint_mask[idx] {
                continue;
            }

            // Collect non-glint neighbors in 5x5 window
            let mut neighbors: Vec<u8> = Vec::with_capacity(25);
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let nidx = ny as usize * width + nx as usize;
                        if !glint_mask[nidx] {
                            neighbors.push(original[nidx]);
                        }
                    }
                }
            }

            if !neighbors.is_empty() {
                neighbors.sort();
                pixels[idx] = neighbors[neighbors.len() / 2];
            }
        }
    }
}

/// Bilinear upscale by 2x.
pub fn upscale_2x(src: &[u8], sw: usize, sh: usize) -> (Vec<u8>, usize, usize) {
    let dw = sw * 2;
    let dh = sh * 2;
    let mut dst = vec![0u8; dw * dh];

    for dy in 0..dh {
        for dx in 0..dw {
            // Source coordinates in float
            let sx = dx as f32 * 0.5;
            let sy = dy as f32 * 0.5;
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let p00 = src[y0 * sw + x0] as f32;
            let p01 = src[y0 * sw + x1] as f32;
            let p10 = src[y1 * sw + x0] as f32;
            let p11 = src[y1 * sw + x1] as f32;

            let v = p00 * (1.0 - fx) * (1.0 - fy)
                + p01 * fx * (1.0 - fy)
                + p10 * (1.0 - fx) * fy
                + p11 * fx * fy;
            dst[dy * dw + dx] = v as u8;
        }
    }

    (dst, dw, dh)
}

/// Extract a rotated ROI aligned to the eye axis (line between two landmark points).
///
/// - `anchor1`, `anchor2`: two points defining the eye axis (e.g., eye corners)
/// - `out_w`, `out_h`: output dimensions
/// - `y_expand`: vertical expansion factor beyond the axis line (e.g., 0.8 = 80% of eye width)
///
/// Returns a grayscale image that is axis-aligned in eye coordinates.
pub fn rotated_eye_roi(
    src: &[u8],
    sw: usize,
    sh: usize,
    anchor1: (f64, f64),
    anchor2: (f64, f64),
    out_w: usize,
    out_h: usize,
    y_expand: f64,
) -> Vec<u8> {
    let cx = (anchor1.0 + anchor2.0) / 2.0;
    let cy = (anchor1.1 + anchor2.1) / 2.0;
    let dx = anchor2.0 - anchor1.0;
    let dy = anchor2.1 - anchor1.1;
    let eye_width = (dx * dx + dy * dy).sqrt();
    if eye_width < 1.0 {
        return vec![0u8; out_w * out_h];
    }
    let angle = dy.atan2(dx);

    // Sample region: 1.4x eye width horizontally, y_expand × eye width vertically
    let sample_w = eye_width * 1.4;
    let sample_h = eye_width * y_expand * 2.0;

    let cos = angle.cos();
    let sin = angle.sin();

    let mut dst = vec![0u8; out_w * out_h];

    for oy in 0..out_h {
        for ox in 0..out_w {
            // Normalized coordinates in output [-1, 1]
            let nx = (ox as f64 / out_w as f64 - 0.5) * 2.0;
            let ny = (oy as f64 / out_h as f64 - 0.5) * 2.0;

            // Local coordinates in eye frame
            let lx = nx * sample_w / 2.0;
            let ly = ny * sample_h / 2.0;

            // Rotate into image frame
            let ix = cx + lx * cos - ly * sin;
            let iy = cy + lx * sin + ly * cos;

            if ix >= 0.0 && iy >= 0.0 && ix < sw as f64 - 1.0 && iy < sh as f64 - 1.0 {
                // Bilinear sample
                let x0 = ix as usize;
                let y0 = iy as usize;
                let fx = (ix - x0 as f64) as f32;
                let fy = (iy - y0 as f64) as f32;
                let p00 = src[y0 * sw + x0] as f32;
                let p01 = src[y0 * sw + x0 + 1] as f32;
                let p10 = src[(y0 + 1) * sw + x0] as f32;
                let p11 = src[(y0 + 1) * sw + x0 + 1] as f32;
                let v = p00 * (1.0 - fx) * (1.0 - fy)
                    + p01 * fx * (1.0 - fy)
                    + p10 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[oy * out_w + ox] = v as u8;
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clahe_expands_contrast() {
        // Low contrast image: values 100-110
        let mut pixels: Vec<u8> = (0..100).map(|i| 100 + (i % 11) as u8).collect();
        clahe_global(&mut pixels, 0.05);
        // After equalization, range should span more
        let min = *pixels.iter().min().unwrap();
        let max = *pixels.iter().max().unwrap();
        assert!((max - min) > 50, "range {}-{}", min, max);
    }

    #[test]
    fn glint_removal_replaces_bright_spots() {
        // 10x10 uniform gray image with one bright spot
        let mut pixels = vec![100u8; 100];
        pixels[55] = 250; // glint at (5,5)
        remove_glints(&mut pixels, 10, 10, 2.0);
        // Should be replaced with local median (100)
        assert!(pixels[55] < 150, "glint not removed: {}", pixels[55]);
    }

    #[test]
    fn upscale_preserves_content() {
        let src = vec![10u8, 20, 30, 40];
        let (dst, dw, dh) = upscale_2x(&src, 2, 2);
        assert_eq!(dw, 4);
        assert_eq!(dh, 4);
        // Corners should roughly match source
        assert!((dst[0] as i32 - 10).abs() < 5);
        assert!((dst[15] as i32 - 40).abs() < 5);
    }

    #[test]
    fn rotated_roi_horizontal() {
        // 20x10 gradient: brightness increases with x
        let sw = 20;
        let sh = 10;
        let src: Vec<u8> = (0..sw * sh).map(|i| ((i % sw) * 12) as u8).collect();
        // Horizontal eye axis from (2, 5) to (18, 5)
        let roi = rotated_eye_roi(&src, sw, sh, (2.0, 5.0), (18.0, 5.0), 40, 20, 0.4);
        assert_eq!(roi.len(), 40 * 20);
        // Middle row: should have gradient from dark to bright
        let mid_y = 10;
        assert!(roi[mid_y * 40] < roi[mid_y * 40 + 35]);
    }
}
