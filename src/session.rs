//! Recorded eye-tracking session for offline benchmarking.
//!
//! Captures calibration samples (features + click targets) and validation
//! samples (features + known target) so different algorithms / parameters
//! can be compared on identical input data.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SACSESS1";

/// One calibration sample: features + the target screen position the user
/// was looking at.
#[derive(Debug, Clone)]
pub struct CalibFrame {
    pub features: Vec<f32>,
    pub target_x: f32,
    pub target_y: f32,
}

/// One validation sample: features only (target is constant for the session).
#[derive(Debug, Clone)]
pub struct ValidFrame {
    pub features: Vec<f32>,
}

/// Recorded session.
#[derive(Debug, Clone)]
pub struct Session {
    pub screen_w: u32,
    pub screen_h: u32,
    pub calibration: Vec<CalibFrame>,
    pub validation: Vec<ValidFrame>,
    pub validation_target: (f32, f32),
}

impl Session {
    pub fn new(screen_w: u32, screen_h: u32, validation_target: (f32, f32)) -> Self {
        Self {
            screen_w,
            screen_h,
            calibration: Vec::new(),
            validation: Vec::new(),
            validation_target,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        w.write_all(MAGIC)?;
        write_u32(&mut w, self.screen_w)?;
        write_u32(&mut w, self.screen_h)?;
        write_f32(&mut w, self.validation_target.0)?;
        write_f32(&mut w, self.validation_target.1)?;
        write_u32(&mut w, self.calibration.len() as u32)?;
        for c in &self.calibration {
            write_features(&mut w, &c.features)?;
            write_f32(&mut w, c.target_x)?;
            write_f32(&mut w, c.target_y)?;
        }
        write_u32(&mut w, self.validation.len() as u32)?;
        for v in &self.validation {
            write_features(&mut w, &v.features)?;
        }
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad magic — not a saccade session file",
            ));
        }
        let screen_w = read_u32(&mut r)?;
        let screen_h = read_u32(&mut r)?;
        let val_x = read_f32(&mut r)?;
        let val_y = read_f32(&mut r)?;
        let n_calib = read_u32(&mut r)? as usize;
        let mut calibration = Vec::with_capacity(n_calib);
        for _ in 0..n_calib {
            let features = read_features(&mut r)?;
            let target_x = read_f32(&mut r)?;
            let target_y = read_f32(&mut r)?;
            calibration.push(CalibFrame { features, target_x, target_y });
        }
        let n_valid = read_u32(&mut r)? as usize;
        let mut validation = Vec::with_capacity(n_valid);
        for _ in 0..n_valid {
            let features = read_features(&mut r)?;
            validation.push(ValidFrame { features });
        }
        Ok(Self {
            screen_w,
            screen_h,
            calibration,
            validation,
            validation_target: (val_x, val_y),
        })
    }
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32<W: Write>(w: &mut W, v: f32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_features<W: Write>(w: &mut W, feats: &[f32]) -> std::io::Result<()> {
    write_u32(w, feats.len() as u32)?;
    for &v in feats {
        write_f32(w, v)?;
    }
    Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_f32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_features<R: Read>(r: &mut R) -> std::io::Result<Vec<f32>> {
    let n = read_u32(r)? as usize;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(read_f32(r)?);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut s = Session::new(1920, 1080, (960.0, 540.0));
        s.calibration.push(CalibFrame {
            features: vec![1.0, 2.0, 3.0],
            target_x: 100.0,
            target_y: 200.0,
        });
        s.calibration.push(CalibFrame {
            features: vec![4.0, 5.0, 6.0],
            target_x: 300.0,
            target_y: 400.0,
        });
        s.validation.push(ValidFrame { features: vec![7.0, 8.0, 9.0] });

        let path = std::env::temp_dir().join("saccade_test_session.bin");
        s.save(&path).unwrap();
        let loaded = Session::load(&path).unwrap();

        assert_eq!(loaded.screen_w, 1920);
        assert_eq!(loaded.screen_h, 1080);
        assert_eq!(loaded.validation_target, (960.0, 540.0));
        assert_eq!(loaded.calibration.len(), 2);
        assert_eq!(loaded.calibration[0].features, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.calibration[1].target_x, 300.0);
        assert_eq!(loaded.validation.len(), 1);
        assert_eq!(loaded.validation[0].features, vec![7.0, 8.0, 9.0]);

        let _ = std::fs::remove_file(&path);
    }
}
