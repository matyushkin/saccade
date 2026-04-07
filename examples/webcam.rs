//! Real-time eye tracking with calibration.
//!
//! Flow:
//! 1. Press SPACE to start 9-point calibration (look at each red dot)
//! 2. After calibration, gaze point shown as white circle on screen
//!
//! Controls:
//!   SPACE — start/restart calibration
//!   ESC   — quit
//!
//! cargo run --release --features demo --example webcam

use display_info::DisplayInfo;
use minifb::{Key, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use saccade::blink::{BlinkDetector, EyeState};
use saccade::calibration::{self, CalibrationSample, GazeMapper, NormalizedPupil};
use saccade::one_euro::OneEuroFilter2D;
use saccade::classify::{EyeEvent, IVTClassifier};
use saccade::frame::GrayFrame;
use saccade::preprocess;
use saccade::timm::{self, TimmConfig};
use std::time::Instant;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const FACE_M: &str = "seeta_fd_frontal_v1.0.bin";
const FACE_U: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";
const PFLD_M: &str = "pfld.onnx";
const PFLD_U: &str = "https://github.com/cunjian/pytorch_face_landmark/raw/refs/heads/master/onnx/pfld.onnx";
const OCEC_M: &str = "ocec_m.onnx";
const OCEC_U: &str = "https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_m.onnx";

fn dl(p: &str, u: &str) {
    if !std::path::Path::new(p).exists() {
        println!("Downloading {p}...");
        let _ = std::process::Command::new("curl").args(["-L","-o",p,u]).status();
    }
}
fn load(p: &str) -> Model {
    tract_onnx::onnx().model_for_path(p).unwrap().into_optimized().unwrap().into_runnable().unwrap()
}

/// App phase
#[derive(PartialEq)]
enum Phase { Live, Calibrating, Calibrated }

/// Smooth pursuit target — slow two-stage trajectory.
/// Stage 1 (0-15s): outer circle covers screen boundary (natural smooth pursuit).
/// Stage 2 (15-30s): inner circle fills the interior for more data points.
/// Total duration: 30 seconds.
fn pursuit_target(t: f64, sw: f64, sh: f64) -> (f64, f64) {
    let mx = sw * 0.12;
    let my = sh * 0.12;
    let cx = sw / 2.0;
    let cy = sh / 2.0;
    let amp_x = (sw - 2.0 * mx) / 2.0;
    let amp_y = (sh - 2.0 * my) / 2.0;

    let two_pi = 2.0 * std::f64::consts::PI;

    if t < 15.0 {
        // Large outer ellipse, one full revolution (15 sec)
        let theta = two_pi * t / 15.0;
        (cx + amp_x * theta.cos(), cy + amp_y * theta.sin())
    } else {
        // Smaller figure-8 for inner coverage (15 sec)
        let t2 = t - 15.0;
        let theta = two_pi * t2 / 15.0;
        let x = cx + (amp_x * 0.5) * (2.0 * theta).sin();
        let y = cy + (amp_y * 0.5) * theta.sin();
        (x, y)
    }
}

fn main() {
    dl(FACE_M, FACE_U); dl(PFLD_M, PFLD_U); dl(OCEC_M, OCEC_U);

    let mut face_det = rustface::create_detector(FACE_M).expect("face model");
    face_det.set_min_face_size(80);
    face_det.set_score_thresh(2.0);
    let pfld = load(PFLD_M);
    let ocec = load(OCEC_M);

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = loop {
        match Camera::new(CameraIndex::Index(0), format.clone()) {
            Ok(c) => break c,
            Err(e) => {
                eprintln!("Camera init failed: {e}. Retrying in 2s...");
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    };
    camera.open_stream().expect("stream");
    let cam_res = camera.resolution();
    // Process at higher resolution for better pupil precision
    let sd = (cam_res.width() / 720).max(1) as usize;
    let cw = cam_res.width() as usize / sd;
    let ch = cam_res.height() as usize / sd;
    println!("Camera: {}x{} -> {}x{}", cam_res.width(), cam_res.height(), cw, ch);

    // Get screen size from primary display
    let displays = DisplayInfo::all().expect("failed to get displays");
    let primary = displays.iter().find(|d| d.is_primary).unwrap_or(&displays[0]);
    let sw = primary.width as usize;
    let sh = primary.height as usize;
    println!("Screen: {sw}x{sh}");

    // Picture-in-picture: small camera preview in corner
    let pip_w = 240usize;
    let pip_h = pip_w * ch / cw;

    let mut window = Window::new(
        "Saccade Eye Tracker [SPACE=calibrate, ESC=quit]",
        sw, sh,
        WindowOptions { borderless: true, topmost: true, ..WindowOptions::default() },
    ).unwrap();
    window.set_position(0, 0);
    window.set_target_fps(60);

    let tcfg = TimmConfig { gradient_threshold: 0.2, use_weight_map: true, weight_blur_sigma: 2.0 };
    let mut face_sm = SmRect::new(0.25);
    let mut lp_sm = SmPt::new(0.7);
    let mut rp_sm = SmPt::new(0.7);
    let mut l_open_sm = 1.0f64;
    let mut r_open_sm = 1.0f64;
    let mut l_blink = BlinkDetector::new(); l_blink.confidence_threshold = 0.15;
    let mut r_blink = BlinkDetector::new(); r_blink.confidence_threshold = 0.15;
    let mut l_roi = EyeRoi::new();
    let mut r_roi = EyeRoi::new();
    let mut bl_cal_sum = (0.0f64, 0.0f64);
    let mut bl_cal_n = 0u32;
    let mut classifier = IVTClassifier::default_params();
    let start = Instant::now();
    let mut buf = vec![0u32; sw * sh];
    let mut fps = FpsC::new();
    let mut no_face = 0u32;

    // Skip heavy face detection on most frames
    let mut frame_n = 0u64;

    // Smooth pursuit calibration state
    let mut phase = Phase::Live;
    let mut gaze_mapper = GazeMapper::new();
    let mut calib_samples: Vec<CalibrationSample> = Vec::new();
    let mut calib_start_ms: u64 = 0;
    let calib_duration_ms: u64 = 30_000; // 30 seconds smooth pursuit

    // 1€ filter for gaze cursor — adaptive low-pass
    // min_cutoff=1.0 Hz (smooths jitter at rest), beta=0.1 (responsive to fast motion)
    let mut gaze_filter = OneEuroFilter2D::new(1.0, 0.1, 1.0);

    println!("Press SPACE to start calibration. ESC to quit.");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_ms = start.elapsed().as_millis() as u64;

        // Handle SPACE key
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            phase = Phase::Calibrating;
            calib_samples.clear();
            calib_start_ms = now_ms;
            gaze_mapper = GazeMapper::new();
            gaze_filter.reset();
            println!("\nSmooth pursuit calibration: follow the green dot for 30 seconds.");
        }

        // Capture
        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(i) => i, Err(_) => continue },
            Err(_) => continue,
        };
        let (fw, fh) = (decoded.width() as usize, decoded.height() as usize);
        let rf = decoded.as_raw();
        let mut rgb = vec![0u8; cw*ch*3];
        let mut gray = vec![0u8; cw*ch];
        for y in 0..ch { for x in 0..cw {
            let (sx, sy) = (x*sd, y*sd);
            if sx<fw && sy<fh {
                let si=(sy*fw+sx)*3; let di=(y*cw+x)*3;
                rgb[di]=rf[si]; rgb[di+1]=rf[si+1]; rgb[di+2]=rf[si+2];
                gray[y*cw+x]=(0.299*rf[si] as f32+0.587*rf[si+1] as f32+0.114*rf[si+2] as f32) as u8;
            }
        }}
        frame_n += 1;
        // Run face detection only every 3 frames
        let do_face_det = frame_n % 3 == 1 || face_sm.init == false;
        let faces = if do_face_det {
            face_det.detect(&ImageData::new(&gray, cw as u32, ch as u32))
        } else {
            Vec::new()
        };

        // Clear screen to black
        for px in buf.iter_mut() { *px = 0; }

        // Render camera preview as picture-in-picture in top-right corner
        // Only show in LIVE/Calibrated mode (hidden during calibration to avoid distraction)
        if phase != Phase::Calibrating {
            let pip_x = sw - pip_w - 10;
            let pip_y = 10;
            for py in 0..pip_h { for px in 0..pip_w {
                let sx = px * cw / pip_w;
                let sy = py * ch / pip_h;
                if sx < cw && sy < ch {
                    let si = (sy * cw + sx) * 3;
                    let r = rgb[si] as u32;
                    let g = rgb[si+1] as u32;
                    let b = rgb[si+2] as u32;
                    let dx = pip_x + px;
                    let dy = pip_y + py;
                    if dx < sw && dy < sh {
                        buf[dy * sw + dx] = (r<<16)|(g<<8)|b;
                    }
                }
            }}
            // PIP border
            draw_rect(&mut buf, sw, sh, pip_x.saturating_sub(1), pip_y.saturating_sub(1), pip_w+2, pip_h+2, 0xFFFFFF);
        }

        let detected = faces.iter().max_by_key(|f| { let b=f.bbox(); b.width()*b.height() });
        // Use detected face OR fall back to smoothed previous face
        let have_face = detected.is_some() || (face_sm.init && !do_face_det);

        // Track pupils and eye corner landmarks
        let mut left_pupil: Option<(f64,f64)> = None;
        let mut right_pupil: Option<(f64,f64)> = None;
        // Eye corners: (outer_corner, inner_corner)
        let mut left_corners: Option<((f64,f64),(f64,f64))> = None;
        let mut right_corners: Option<((f64,f64),(f64,f64))> = None;

        if have_face {
            no_face = 0;
            if let Some(face) = detected {
                let bb = face.bbox();
                face_sm.update(bb.x().max(0) as f64, bb.y().max(0) as f64, bb.width() as f64, bb.height() as f64);
            }
            let (fx,fy,fwf,fhf) = face_sm.get();

            if let Some(lm) = run_pfld(&pfld, &gray, cw, ch, fx, fy, fwf, fhf) {
                // OCEC blink
                let r_op = run_ocec(&ocec, &gray, cw, ch, &lm, 36, 41, &mut r_roi);
                let l_op = run_ocec(&ocec, &gray, cw, ch, &lm, 42, 47, &mut l_roi);
                l_open_sm = 0.5*l_op as f64 + 0.5*l_open_sm;
                r_open_sm = 0.5*r_op as f64 + 0.5*r_open_sm;

                if bl_cal_n < 45 {
                    bl_cal_sum.0 += l_open_sm; bl_cal_sum.1 += r_open_sm; bl_cal_n += 1;
                    if bl_cal_n == 45 {
                        l_blink.confidence_threshold = (bl_cal_sum.0/45.0*0.4).max(0.03);
                        r_blink.confidence_threshold = (bl_cal_sum.1/45.0*0.4).max(0.03);
                    }
                }
                let l_st = l_blink.update(l_open_sm, now_ms);
                let r_st = r_blink.update(r_open_sm, now_ms);
                // Eye state indicators on top-left of screen
                draw_filled_circle(&mut buf, sw, sh, 30, 30, 10, state_color(l_st));
                draw_filled_circle(&mut buf, sw, sh, 60, 30, 10, state_color(r_st));

                // Extract eye corner landmarks (iBUG 68 scheme):
                // Right eye: 36 (outer) -- 39 (inner)
                // Left eye: 42 (inner) -- 45 (outer)
                let r_outer = (lm[36].0 as f64, lm[36].1 as f64);
                let r_inner = (lm[39].0 as f64, lm[39].1 as f64);
                let l_inner = (lm[42].0 as f64, lm[42].1 as f64);
                let l_outer = (lm[45].0 as f64, lm[45].1 as f64);

                // Detect pupils
                if let Some((px,py)) = detect_pupil(&lm, 42, 47, &gray, cw, ch, &tcfg) {
                    lp_sm.update(px, py);
                    left_pupil = Some(lp_sm.get_f());
                    left_corners = Some((l_outer, l_inner));
                }
                if let Some((px,py)) = detect_pupil(&lm, 36, 41, &gray, cw, ch, &tcfg) {
                    rp_sm.update(px, py);
                    right_pupil = Some(rp_sm.get_f());
                    right_corners = Some((r_outer, r_inner));
                    classifier.update(px, py, now_ms);
                }
            }
        } else {
            no_face += 1;
            if no_face > 30 { l_blink.reset(); r_blink.reset(); lp_sm=SmPt::new(0.5); rp_sm=SmPt::new(0.5); no_face=0; }
        }

        // Compute PPERV feature (head-pose-invariant pupil position).
        // Average both eyes for robustness.
        let norm_pupil = if let (Some(lp), Some(rp), Some((lo, li)), Some((ro, ri))) =
            (left_pupil, right_pupil, left_corners, right_corners)
        {
            let l_norm = calibration::pperv(lp, lo, li);
            let r_norm = calibration::pperv(rp, ro, ri);
            Some(NormalizedPupil {
                x: (l_norm.x + r_norm.x) / 2.0,
                y: (l_norm.y + r_norm.y) / 2.0,
            })
        } else { None };

        // --- Smooth pursuit calibration phase ---
        if phase == Phase::Calibrating {
            let elapsed_ms = now_ms.saturating_sub(calib_start_ms);
            let t = elapsed_ms as f64 / 1000.0;
            let (tx, ty) = pursuit_target(t, sw as f64, sh as f64);

            // (Target drawn later, after bg_buf is saved)

            // Progress bar at bottom
            let progress = (elapsed_ms as f64 / calib_duration_ms as f64).min(1.0);
            let bar_w = (sw as f64 * 0.6) as usize;
            let bar_x = (sw - bar_w) / 2;
            let bar_y = sh - 40;
            for x in 0..bar_w { for y in 0..8 { buf[(bar_y+y)*sw+bar_x+x] = 0x222222; } }
            let filled = (bar_w as f64 * progress) as usize;
            for x in 0..filled { for y in 0..8 { buf[(bar_y+y)*sw+bar_x+x] = 0x00FF00; } }

            // Collect sample every frame after initial settling
            if elapsed_ms > 500 {
                if let Some(np) = norm_pupil {
                    calib_samples.push(CalibrationSample {
                        pupil: np,
                        screen_x: tx,
                        screen_y: ty,
                    });
                }
            }

            // Finish calibration
            if elapsed_ms >= calib_duration_ms {
                let raw_n = calib_samples.len();
                println!("\nCollected {raw_n} samples, filtering outliers...");

                // Two-pass fit: initial fit, compute residuals, remove outliers, refit
                if raw_n >= 10 && gaze_mapper.calibrate(&calib_samples) {
                    // Compute residuals
                    let residuals: Vec<f64> = calib_samples.iter().map(|s| {
                        let (px, py) = gaze_mapper.map(&s.pupil);
                        ((px - s.screen_x).powi(2) + (py - s.screen_y).powi(2)).sqrt()
                    }).collect();
                    let n = residuals.len() as f64;
                    let mean_r = residuals.iter().sum::<f64>() / n;
                    let std_r = (residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n).sqrt();
                    let threshold = mean_r + 1.5 * std_r;

                    let filtered: Vec<CalibrationSample> = calib_samples.iter()
                        .zip(residuals.iter())
                        .filter(|&(_, r)| *r < threshold)
                        .map(|(s, _)| *s)
                        .collect();

                    println!("  Filtered: {} → {} (threshold={:.1}px)", raw_n, filtered.len(), threshold);

                    // Refit with clean samples
                    if filtered.len() >= 10 && gaze_mapper.calibrate(&filtered) {
                        phase = Phase::Calibrated;
                        println!("Calibration complete!");
                    } else {
                        phase = Phase::Live;
                        println!("Refit failed. Press SPACE to retry.");
                    }
                } else {
                    phase = Phase::Live;
                    println!("Calibration failed. Press SPACE to retry.");
                }
            }
        }

        // --- Calibrated gaze overlay ---
        if phase == Phase::Calibrated {
            if let Some(np) = norm_pupil {
                let (raw_sx, raw_sy) = gaze_mapper.map(&np);
                let rsx = raw_sx.clamp(0.0, sw as f64 - 1.0);
                let rsy = raw_sy.clamp(0.0, sh as f64 - 1.0);
                let t_sec = now_ms as f64 / 1000.0;
                let (gx_f, gy_f) = gaze_filter.filter((rsx, rsy), t_sec);
                let gx = gx_f as usize;
                let gy = gy_f as usize;
                // Big gaze cursor — visible across full screen
                draw_filled_circle(&mut buf, sw, sh, gx, gy, 8, 0x00FF00);
                draw_ring(&mut buf, sw, sh, gx, gy, 20, 0xFFFFFF);
                draw_ring(&mut buf, sw, sh, gx, gy, 21, 0xFFFFFF);
                draw_ring(&mut buf, sw, sh, gx, gy, 32, 0x00FF00);
                draw_cross(&mut buf, sw, sh, gx, gy, 0xFFFFFF);
            }
        }

        // FPS bar
        fps.tick();
        let f = fps.fps();
        let bar = (f as usize * 5).min(sw);
        let bc = if f > 15.0 { 0x00FF00 } else if f > 8.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar { for y in 0..4 { buf[y*sw+x] = bc; } }

        // Save background buffer (without the moving target)
        // for inter-frame re-rendering at higher rate
        let bg_buf = buf.clone();

        // Draw current target on top of bg and push to window
        if phase == Phase::Calibrating {
            let elapsed_ms = now_ms.saturating_sub(calib_start_ms);
            let t = elapsed_ms as f64 / 1000.0;
            let (tx, ty) = pursuit_target(t, sw as f64, sh as f64);
            let pulse = ((t * 4.0).sin() * 0.5 + 0.5) * 4.0;
            let r = (14.0 + pulse) as usize;
            draw_filled_circle(&mut buf, sw, sh, tx as usize, ty as usize, r, 0x00FF00);
            draw_ring(&mut buf, sw, sh, tx as usize, ty as usize, r + 4, 0xFFFFFF);
            draw_filled_circle(&mut buf, sw, sh, tx as usize, ty as usize, 3, 0xFFFFFF);
        }
        window.update_with_buffer(&buf, sw, sh).unwrap();

        // Inter-frame refresh: redraw moving target multiple times for smooth pursuit
        if phase == Phase::Calibrating {
            for _ in 0..3 {
                std::thread::sleep(std::time::Duration::from_millis(33));
                let inter_ms = start.elapsed().as_millis() as u64;
                let t2 = inter_ms.saturating_sub(calib_start_ms) as f64 / 1000.0;
                if t2 * 1000.0 >= calib_duration_ms as f64 { break; }
                let (tx2, ty2) = pursuit_target(t2, sw as f64, sh as f64);
                // Copy background
                buf.copy_from_slice(&bg_buf);
                // Redraw target with new position
                let pulse = ((t2 * 4.0).sin() * 0.5 + 0.5) * 4.0;
                let r = (14.0 + pulse) as usize;
                draw_filled_circle(&mut buf, sw, sh, tx2 as usize, ty2 as usize, r, 0x00FF00);
                draw_ring(&mut buf, sw, sh, tx2 as usize, ty2 as usize, r + 4, 0xFFFFFF);
                draw_filled_circle(&mut buf, sw, sh, tx2 as usize, ty2 as usize, 3, 0xFFFFFF);
                // Also collect samples during inter-frame (using cached norm_pupil)
                if inter_ms.saturating_sub(calib_start_ms) > 500 {
                    if let Some(np) = norm_pupil {
                        calib_samples.push(CalibrationSample {
                            pupil: np,
                            screen_x: tx2,
                            screen_y: ty2,
                        });
                    }
                }
                window.update_with_buffer(&buf, sw, sh).unwrap();
            }
        }

        if fps.count % 30 == 0 {
            let bl = l_blink.blink_count() + r_blink.blink_count();
            let bpm = (l_blink.blinks_per_minute(now_ms,60000)+r_blink.blinks_per_minute(now_ms,60000))/2.0;
            let fix = classifier.events().iter().filter(|e| matches!(e, EyeEvent::Fixation(_))).count();
            let sac = classifier.events().iter().filter(|e| matches!(e, EyeEvent::Saccade(_))).count();
            let mode = match phase { Phase::Live => "LIVE", Phase::Calibrating => "CALIB", Phase::Calibrated => "GAZE" };
            print!("\r[{mode}] FPS:{f:.0} | Blinks:{bl}({bpm:.0}/m) | Fix:{fix} Sac:{sac}    ");
        }
    }
    println!("\nDone.");
}

// --- Eye ROI info from landmarks ---
fn eye_roi_info(lm: &[(f32,f32)], si: usize, ei: usize) -> ((f64,f64),(f64,f64)) {
    let pts = &lm[si..=ei];
    let cx = pts.iter().map(|p| p.0 as f64).sum::<f64>() / pts.len() as f64;
    let cy = pts.iter().map(|p| p.1 as f64).sum::<f64>() / pts.len() as f64;
    let min_x = pts.iter().map(|p| p.0 as f64).fold(f64::MAX, f64::min);
    let max_x = pts.iter().map(|p| p.0 as f64).fold(f64::MIN, f64::max);
    let min_y = pts.iter().map(|p| p.1 as f64).fold(f64::MAX, f64::min);
    let max_y = pts.iter().map(|p| p.1 as f64).fold(f64::MIN, f64::max);
    ((cx, cy), ((max_x-min_x).max(1.0), (max_y-min_y).max(1.0)))
}

// --- Tract runners ---
fn run_pfld(m: &Model, g: &[u8], w: usize, h: usize, fx: usize, fy: usize, fw: usize, fh: usize) -> Option<Vec<(f32,f32)>> {
    if fx+fw>w||fy+fh>h||fw<20||fh<20 { return None; }
    let s=112; let mut d=vec![0.0f32;3*s*s];
    for y in 0..s{for x in 0..s{let(sx,sy)=(fx+x*fw/s,fy+y*fh/s);let v=if sx<w&&sy<h{g[sy*w+sx]as f32/255.0}else{0.0};d[y*s+x]=v;d[s*s+y*s+x]=v;d[2*s*s+y*s+x]=v;}}
    let t=Tensor::from(tract_ndarray::Array4::from_shape_vec((1,3,s,s),d).ok()?).into();
    let r=m.run(tvec![t]).ok()?; let o=r[0].to_array_view::<f32>().ok()?; let f=o.as_slice()?;
    if f.len()<136{return None;} Some((0..68).map(|i|(f[i*2]*fw as f32+fx as f32,f[i*2+1]*fh as f32+fy as f32)).collect())
}

fn run_ocec(m: &Model, g: &[u8], w: usize, h: usize, lm: &[(f32,f32)], si: usize, ei: usize, rs: &mut EyeRoi) -> f32 {
    let pts=&lm[si..=ei];
    let(mnx,mxx)=(pts.iter().map(|p|p.0).fold(f32::MAX,f32::min),pts.iter().map(|p|p.0).fold(f32::MIN,f32::max));
    let(mny,mxy)=(pts.iter().map(|p|p.1).fold(f32::MAX,f32::min),pts.iter().map(|p|p.1).fold(f32::MIN,f32::max));
    let(mx,my)=((mxx-mnx)*0.15,(mxy-mny)*0.3);
    if!rs.frozen{rs.update((mnx-mx).max(0.0)as f64,(mny-my).max(0.0)as f64,((mxx-mnx)+2.0*mx).max(4.0)as f64,((mxy-mny)+2.0*my).max(4.0)as f64);}
    let(rx,ry,rw,rh)=rs.get(); if rx+rw>w||ry+rh>h{return 1.0;}
    let(oh,ow)=(24,40); let mut d=vec![0.0f32;3*oh*ow];
    for y in 0..oh{for x in 0..ow{let(sx,sy)=(rx+x*rw/ow,ry+y*rh/oh);let v=if sx<w&&sy<h{g[sy*w+sx]as f32/255.0}else{0.0};d[y*ow+x]=v;d[oh*ow+y*ow+x]=v;d[2*oh*ow+y*ow+x]=v;}}
    let t=Tensor::from(tract_ndarray::Array4::from_shape_vec((1,3,oh,ow),d).unwrap()).into();
    let r=match m.run(tvec![t]){Ok(r)=>r,Err(_)=>return 1.0};
    let v=r[0].to_array_view::<f32>().ok().and_then(|a|a.as_slice().map(|s|if s.is_empty(){1.0}else{s[0]})).unwrap_or(1.0);
    if v<0.3{rs.freeze();}else{rs.unfreeze();} v
}

/// Detect pupil center using axis-aligned eye ROI + contrast stretch + Timm & Barth.
fn detect_pupil(lm:&[(f32,f32)],si:usize,ei:usize,g:&[u8],w:usize,h:usize,cfg:&TimmConfig)->Option<(f64,f64)>{
    let pts=&lm[si..=ei];
    let(mnx,mxx)=(pts.iter().map(|p|p.0).fold(f32::MAX,f32::min),pts.iter().map(|p|p.0).fold(f32::MIN,f32::max));
    let(mny,mxy)=(pts.iter().map(|p|p.1).fold(f32::MAX,f32::min),pts.iter().map(|p|p.1).fold(f32::MIN,f32::max));
    let(mx,my)=((mxx-mnx)*0.3,(mxy-mny)*0.5);
    let(rx,ry)=((mnx-mx).max(0.0)as usize,(mny-my).max(0.0)as usize);
    let(rw,rh)=(((mxx-mnx)+2.0*mx)as usize,((mxy-mny)+2.0*my)as usize);
    if rx+rw>w||ry+rh>h||rw<8||rh<6{return None;}
    let mut d=vec![0u8;rw*rh];
    for y in 0..rh{d[y*rw..(y+1)*rw].copy_from_slice(&g[(ry+y)*w+rx..(ry+y)*w+rx+rw]);}
    // Contrast stretch
    let(mn,mx_v)=(*d.iter().min()?,*d.iter().max()?);
    if mx_v>mn+10{let r=(mx_v-mn)as f32;for p in&mut d{*p=((*p as f32-mn as f32)/r*255.0)as u8;}}
    let f=GrayFrame::new(rw as u32,rh as u32,&d);
    let r=timm::detect_center(&f,cfg);
    Some((rx as f64+r.x,ry as f64+r.y))
}

// --- UI helpers ---
struct EyeRoi{rx:f64,ry:f64,rw:f64,rh:f64,init:bool,frozen:bool}
impl EyeRoi{fn new()->Self{Self{rx:0.0,ry:0.0,rw:0.0,rh:0.0,init:false,frozen:false}}fn update(&mut self,rx:f64,ry:f64,rw:f64,rh:f64){if!self.init{self.rx=rx;self.ry=ry;self.rw=rw;self.rh=rh;self.init=true;}else{let a=0.3;self.rx=a*rx+(1.0-a)*self.rx;self.ry=a*ry+(1.0-a)*self.ry;self.rw=a*rw+(1.0-a)*self.rw;self.rh=a*rh+(1.0-a)*self.rh;}}fn get(&self)->(usize,usize,usize,usize){(self.rx.round()as usize,self.ry.round()as usize,self.rw.round().max(4.0)as usize,self.rh.round().max(4.0)as usize)}fn freeze(&mut self){self.frozen=true;}fn unfreeze(&mut self){self.frozen=false;}}
struct SmRect{x:f64,y:f64,w:f64,h:f64,a:f64,init:bool}
impl SmRect{fn new(a:f64)->Self{Self{x:0.0,y:0.0,w:0.0,h:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64,w:f64,h:f64){if!self.init{self.x=x;self.y=y;self.w=w;self.h=h;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;self.w=a*w+(1.0-a)*self.w;self.h=a*h+(1.0-a)*self.h;}}fn get(&self)->(usize,usize,usize,usize){(self.x.round()as usize,self.y.round()as usize,self.w.round().max(1.0)as usize,self.h.round().max(1.0)as usize)}}
struct SmPt{x:f64,y:f64,a:f64,init:bool}
impl SmPt{fn new(a:f64)->Self{Self{x:0.0,y:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64){if!self.init{self.x=x;self.y=y;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;}}fn get(&self)->(usize,usize){(self.x.round()as usize,self.y.round()as usize)}fn get_f(&self)->(f64,f64){(self.x,self.y)}}
fn state_color(s:EyeState)->u32{match s{EyeState::Open=>0x00FF00,EyeState::Blinking=>0xFFFF00,EyeState::Closed=>0xFF0000}}
fn draw_filled_circle(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for dy in 0..=r{for dx in 0..=r{if dx*dx+dy*dy<=r*r{for&(sx,sy)in&[(cx+dx,cy+dy),(cx.wrapping_sub(dx),cy+dy),(cx+dx,cy.wrapping_sub(dy)),(cx.wrapping_sub(dx),cy.wrapping_sub(dy))]{if sx<w&&sy<h{b[sy*w+sx]=c;}}}}}}
fn draw_ring(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for i in 0..64{let t=2.0*std::f64::consts::PI*i as f64/64.0;let x=(cx as f64+r as f64*t.cos()).round()as i32;let y=(cy as f64+r as f64*t.sin()).round()as i32;if x>=0&&(x as usize)<w&&y>=0&&(y as usize)<h{b[y as usize*w+x as usize]=c;}}}
fn draw_rect(b:&mut[u32],w:usize,h:usize,x:usize,y:usize,rw:usize,rh:usize,c:u32){for dx in 0..rw{let px=x+dx;if px<w{if y<h{b[y*w+px]=c;}let by=y+rh.saturating_sub(1);if by<h{b[by*w+px]=c;}}}for dy in 0..rh{let py=y+dy;if py<h{if x<w{b[py*w+x]=c;}let bx=x+rw.saturating_sub(1);if bx<w{b[py*w+bx]=c;}}}}
fn draw_cross(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,c:u32){for d in -6i32..=6{let x=cx as i32+d;if x>=0&&(x as usize)<w&&cy<h{b[cy*w+x as usize]=c;}let y=cy as i32+d;if y>=0&&(y as usize)<h&&cx<w{b[y as usize*w+cx]=c;}}}
struct FpsC{t:Instant,count:u64,fps:f64}
impl FpsC{fn new()->Self{Self{t:Instant::now(),count:0,fps:0.0}}fn tick(&mut self){self.count+=1;let e=self.t.elapsed().as_secs_f64();if e>=1.0{self.fps=self.count as f64/e;self.count=0;self.t=Instant::now();}}fn fps(&self)->f64{self.fps}}
