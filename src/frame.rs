//! Generic frame input trait with adapters for different image sources.

/// Pixel format of the frame data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Single channel, 8-bit grayscale.
    Gray8,
    /// 3 channels, 8-bit RGB.
    Rgb8,
    /// 3 channels, 8-bit BGR (common in OpenCV).
    Bgr8,
    /// 4 channels, 8-bit RGBA.
    Rgba8,
}

/// A rectangular region of interest within a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Trait for image frame input. Implementors provide grayscale pixel access.
pub trait Frame {
    /// Width of the frame in pixels.
    fn width(&self) -> u32;
    /// Height of the frame in pixels.
    fn height(&self) -> u32;
    /// Access to grayscale pixel data (row-major, 1 byte per pixel).
    fn gray_pixels(&self) -> &[u8];

    /// Total number of pixels.
    fn pixel_count(&self) -> usize {
        self.width() as usize * self.height() as usize
    }

    /// Get a single grayscale pixel value. Returns 0 if out of bounds.
    fn pixel_at(&self, x: u32, y: u32) -> u8 {
        if x < self.width() && y < self.height() {
            self.gray_pixels()[(y as usize) * (self.width() as usize) + (x as usize)]
        } else {
            0
        }
    }
}

/// A frame backed by a raw grayscale pixel buffer.
pub struct GrayFrame<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> GrayFrame<'a> {
    /// Create a new grayscale frame from raw pixel data.
    ///
    /// # Panics
    /// Panics if `data.len() != width * height`.
    pub fn new(width: u32, height: u32, data: &'a [u8]) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "data length must equal width * height"
        );
        Self { width, height, data }
    }
}

impl Frame for GrayFrame<'_> {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn gray_pixels(&self) -> &[u8] {
        self.data
    }
}

/// An owned grayscale frame.
pub struct OwnedGrayFrame {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl OwnedGrayFrame {
    /// Create from owned data.
    ///
    /// # Panics
    /// Panics if `data.len() != width * height`.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "data length must equal width * height"
        );
        Self { width, height, data }
    }

    /// Create by converting RGB data to grayscale.
    ///
    /// # Panics
    /// Panics if `rgb_data.len() != width * height * 3`.
    pub fn from_rgb(width: u32, height: u32, rgb_data: &[u8]) -> Self {
        let npixels = (width as usize) * (height as usize);
        assert_eq!(rgb_data.len(), npixels * 3, "rgb_data length must equal width * height * 3");
        let data: Vec<u8> = rgb_data
            .chunks_exact(3)
            .map(|rgb| {
                // ITU-R BT.601 luma coefficients
                (0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32) as u8
            })
            .collect();
        Self { width, height, data }
    }

    /// Create by converting RGBA data to grayscale (alpha ignored).
    ///
    /// # Panics
    /// Panics if `rgba_data.len() != width * height * 4`.
    pub fn from_rgba(width: u32, height: u32, rgba_data: &[u8]) -> Self {
        let npixels = (width as usize) * (height as usize);
        assert_eq!(
            rgba_data.len(),
            npixels * 4,
            "rgba_data length must equal width * height * 4"
        );
        let data: Vec<u8> = rgba_data
            .chunks_exact(4)
            .map(|rgba| {
                (0.299 * rgba[0] as f32 + 0.587 * rgba[1] as f32 + 0.114 * rgba[2] as f32) as u8
            })
            .collect();
        Self { width, height, data }
    }

    /// Extract a sub-frame from a region of interest. Clamps ROI to frame bounds.
    pub fn crop(frame: &dyn Frame, roi: Roi) -> Self {
        let x0 = roi.x.min(frame.width());
        let y0 = roi.y.min(frame.height());
        let x1 = (roi.x + roi.width).min(frame.width());
        let y1 = (roi.y + roi.height).min(frame.height());
        let w = x1 - x0;
        let h = y1 - y0;
        let src = frame.gray_pixels();
        let stride = frame.width() as usize;
        let mut data = Vec::with_capacity((w as usize) * (h as usize));
        for row in y0..y1 {
            let start = (row as usize) * stride + (x0 as usize);
            data.extend_from_slice(&src[start..start + w as usize]);
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }

    /// Downscale by an integer factor using area averaging.
    pub fn downscale(frame: &dyn Frame, factor: u32) -> Self {
        assert!(factor > 0, "downscale factor must be > 0");
        if factor == 1 {
            return Self {
                width: frame.width(),
                height: frame.height(),
                data: frame.gray_pixels().to_vec(),
            };
        }
        let new_w = frame.width() / factor;
        let new_h = frame.height() / factor;
        let src = frame.gray_pixels();
        let stride = frame.width() as usize;
        let area = (factor * factor) as u32;
        let mut data = Vec::with_capacity((new_w as usize) * (new_h as usize));
        for ny in 0..new_h {
            for nx in 0..new_w {
                let mut sum: u32 = 0;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let sx = (nx * factor + dx) as usize;
                        let sy = (ny * factor + dy) as usize;
                        sum += src[sy * stride + sx] as u32;
                    }
                }
                data.push((sum / area) as u8);
            }
        }
        Self {
            width: new_w,
            height: new_h,
            data,
        }
    }
}

impl Frame for OwnedGrayFrame {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn gray_pixels(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_frame_basics() {
        let data = vec![10, 20, 30, 40, 50, 60];
        let frame = GrayFrame::new(3, 2, &data);
        assert_eq!(frame.width(), 3);
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.pixel_count(), 6);
        assert_eq!(frame.pixel_at(0, 0), 10);
        assert_eq!(frame.pixel_at(2, 1), 60);
        assert_eq!(frame.pixel_at(3, 0), 0); // out of bounds
    }

    #[test]
    fn from_rgb_conversion() {
        // Pure red, pure green, pure blue, white
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let frame = OwnedGrayFrame::from_rgb(2, 2, &rgb);
        assert_eq!(frame.pixel_at(0, 0), 76); // 0.299 * 255 ≈ 76
        assert_eq!(frame.pixel_at(1, 0), 149); // 0.587 * 255 ≈ 149
        assert_eq!(frame.pixel_at(0, 1), 29); // 0.114 * 255 ≈ 29
        assert!(frame.pixel_at(1, 1) >= 254); // ~255 (rounding)
    }

    #[test]
    fn crop_frame() {
        let data: Vec<u8> = (0..16).collect();
        let frame = GrayFrame::new(4, 4, &data);
        let cropped = OwnedGrayFrame::crop(&frame, Roi { x: 1, y: 1, width: 2, height: 2 });
        assert_eq!(cropped.width(), 2);
        assert_eq!(cropped.height(), 2);
        assert_eq!(cropped.gray_pixels(), &[5, 6, 9, 10]);
    }

    #[test]
    fn downscale_frame() {
        // 4×4 frame, downscale by 2 → 2×2
        let data: Vec<u8> = vec![
            10, 20, 30, 40,
            10, 20, 30, 40,
            50, 60, 70, 80,
            50, 60, 70, 80,
        ];
        let frame = GrayFrame::new(4, 4, &data);
        let small = OwnedGrayFrame::downscale(&frame, 2);
        assert_eq!(small.width(), 2);
        assert_eq!(small.height(), 2);
        assert_eq!(small.pixel_at(0, 0), 15); // avg(10,20,10,20)
        assert_eq!(small.pixel_at(1, 0), 35); // avg(30,40,30,40)
        assert_eq!(small.pixel_at(0, 1), 55); // avg(50,60,50,60)
        assert_eq!(small.pixel_at(1, 1), 75); // avg(70,80,70,80)
    }
}
