#!/usr/bin/env python3
"""
Ablation study: compare feature strategies on MPIIGaze Original data.

Variants:
  pixel_sm   : 10×6 hist-eq per eye, 120-D  (≈ current Rust baseline)
  pixel_lg   : 20×12 hist-eq per eye, 480-D
  cnn_only   : MobileGaze (yaw, pitch), 2-D
  hybrid     : pixel_sm + CNN angles, 122-D
  hybrid_lg  : pixel_lg + CNN angles, 482-D

Usage:
  python3 tools/mpii_ablation.py ./MPIIGaze [--n-calib 200] [--subjects 5] [--stride 5]
"""

import sys, os, argparse, time
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

LAMBDA_CANDIDATES = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
GAZE_ONNX = str(Path(__file__).parent.parent / "mobileone_s0_gaze.onnx")

# ── Fast feature extraction ───────────────────────────────────────────────────

def hist_eq_resize(eye_rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Global histogram equalization + bilinear resize → float32 vector."""
    gray = np.dot(eye_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    pil = Image.fromarray(gray)
    eq  = ImageOps.equalize(pil)
    resized = eq.resize((out_w, out_h), Image.BILINEAR)
    return np.array(resized, dtype=np.float32).ravel() / 255.0


# ── CNN ───────────────────────────────────────────────────────────────────────

_sess = None

def get_sess():
    global _sess
    if _sess is None and HAS_ONNX and os.path.exists(GAZE_ONNX):
        _sess = ort.InferenceSession(GAZE_ONNX, providers=['CPUExecutionProvider'])
    return _sess


def decode_l2cs(bins: np.ndarray) -> float:
    e = np.exp(bins - bins.max())
    e /= e.sum()
    deg = np.linspace(-42, 42, len(bins))
    return float(np.dot(e, deg))


def cnn_infer(face_rgb: np.ndarray):
    sess = get_sess()
    if sess is None:
        return None
    pil = Image.fromarray(face_rgb).resize((448, 448), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    inp = arr.transpose(2, 0, 1)[None].astype(np.float32)
    yaw_b, pitch_b = sess.run(None, {'input': inp})
    return np.deg2rad(decode_l2cs(yaw_b[0])), np.deg2rad(decode_l2cs(pitch_b[0]))


# ── Geometry helpers ──────────────────────────────────────────────────────────

def eye_crop(img, lm6, face_w):
    """Crop eye region from full image using 6 landmarks (12 values)."""
    H, W = img.shape[:2]
    xs, ys = lm6[0::2], lm6[1::2]
    cx, cy = xs.mean(), ys.mean()
    pw = max(8, int(face_w * 0.28))
    ph = max(6, int(face_w * 0.13))
    x0 = max(0, int(cx - pw // 2));  x1 = min(W, x0 + pw)
    y0 = max(0, int(cy - ph // 2));  y1 = min(H, y0 + ph)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return img[y0:y1, x0:x1]


def face_crop(img, lm24):
    """Approximate face bounding box from all 24 eye landmarks."""
    H, W = img.shape[:2]
    xs, ys = lm24[0::2], lm24[1::2]
    re_cx = xs[:6].mean();  le_cx = xs[6:12].mean()
    re_cy = ys[:6].mean();  le_cy = ys[6:12].mean()
    ied = np.hypot(le_cx - re_cx, le_cy - re_cy)
    if ied < 10:
        return None
    fw = int(ied * 3.5)
    cx, cy = int((re_cx + le_cx) / 2), int((re_cy + le_cy) / 2)
    x0 = max(0, cx - fw // 2);  x1 = min(W, x0 + fw)
    y0 = max(0, cy - fw // 2);  y1 = min(H, y0 + fw)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return img[y0:y1, x0:x1]


def gaze_vec_from_annotation(cols):
    """3D gaze direction from annotation columns (0-indexed)."""
    tgt = np.array(cols[26:29], dtype=np.float64)
    re  = np.array(cols[35:38], dtype=np.float64)
    le  = np.array(cols[38:41], dtype=np.float64)
    eye = (re + le) / 2.0
    gv  = tgt - eye
    n   = np.linalg.norm(gv)
    return gv / n if n > 1e-6 else None


def vec_to_yaw_pitch(gv):
    yaw   = np.arctan2(-gv[0], -gv[2])
    pitch = np.arcsin(np.clip(-gv[1], -1, 1))
    return yaw, pitch


def gaze_from_angles(yaw, pitch):
    """Inverse of vec_to_yaw_pitch: yaw=atan2(-gx,-gz), pitch=asin(-gy)."""
    x = -np.sin(yaw) * np.cos(pitch)
    y = -np.sin(pitch)
    z = -np.cos(yaw) * np.cos(pitch)
    n = np.sqrt(x*x + y*y + z*z)
    return np.array([x/n, y/n, z/n])


def angular_error(pred_yaw, pred_pitch, true_vec):
    pv = gaze_from_angles(pred_yaw, pred_pitch)
    return np.degrees(np.arccos(np.clip(np.dot(pv, true_vec), -1, 1)))


# ── Ridge (numpy) ─────────────────────────────────────────────────────────────

def ridge_fit(X, y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def loo_rmse(X, y, lam):
    A = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
    yhat = X @ (A @ X.T @ y)
    h = np.einsum('ij,jk,ik->i', X, A, X)
    r = (y - yhat) / np.maximum(1 - h, 1e-6)
    return float(np.sqrt(np.mean(r**2)))


def best_lambda(X, y):
    return min(LAMBDA_CANDIDATES, key=lambda l: loo_rmse(X, y, l))


# ── Load one subject ──────────────────────────────────────────────────────────

def load_subject(subj_dir, n_calib, need_cnn, stride):
    """Returns list of sample dicts."""
    samples = []
    img_idx = 0

    for day_dir in sorted(subj_dir.iterdir()):
        if not day_dir.is_dir() or day_dir.name == 'Calibration':
            continue
        ann_file = day_dir / 'annotation.txt'
        if not ann_file.exists():
            continue
        lines = [l.split() for l in ann_file.read_text().splitlines() if l.strip()]

        for i, cols in enumerate(lines):
            if len(cols) < 41:
                continue
            img_idx += 1
            # Always include calib frames; stride the rest
            if img_idx > n_calib and (img_idx % stride) != 0:
                continue

            img_path = day_dir / f"{i+1:04d}.jpg"
            if not img_path.exists():
                continue

            gv = gaze_vec_from_annotation(cols)
            if gv is None:
                continue

            img = np.array(Image.open(img_path))
            lm  = np.array(cols[0:24], dtype=np.float32)
            face_w_est = (lm[0::2][6:12].mean() - lm[0::2][:6].mean()) * 3.5

            re = eye_crop(img, lm[0:12],  face_w_est)
            le = eye_crop(img, lm[12:24], face_w_est)
            if re is None or le is None:
                continue

            s = {'gv': gv}
            s['pixel_sm'] = np.concatenate([
                hist_eq_resize(re, 10, 6),
                hist_eq_resize(le, 10, 6)
            ])
            s['pixel_lg'] = np.concatenate([
                hist_eq_resize(re, 20, 12),
                hist_eq_resize(le, 20, 12)
            ])

            if need_cnn:
                fc = face_crop(img, lm)
                if fc is not None:
                    r = cnn_infer(fc)
                    if r is not None:
                        s['cnn'] = np.array(r, dtype=np.float32)

            samples.append(s)
    return samples


# ── Evaluate one mode ─────────────────────────────────────────────────────────

def run_mode(samples, mode, n_calib):
    def feat(s):
        if mode == 'pixel_sm':  return s.get('pixel_sm')
        if mode == 'pixel_lg':  return s.get('pixel_lg')
        if mode == 'cnn_only':  return s.get('cnn')
        if mode == 'hybrid':
            p, c = s.get('pixel_sm'), s.get('cnn')
            return np.concatenate([p, c]) if p is not None and c is not None else None
        if mode == 'hybrid_lg':
            p, c = s.get('pixel_lg'), s.get('cnn')
            return np.concatenate([p, c]) if p is not None and c is not None else None
        return None

    rows, yaws, pitches, gvecs = [], [], [], []
    for s in samples:
        f = feat(s)
        if f is None: continue
        yaw, pitch = vec_to_yaw_pitch(s['gv'])
        rows.append(f); yaws.append(yaw); pitches.append(pitch); gvecs.append(s['gv'])

    if len(rows) < n_calib + 5:
        return None

    Xc = np.array(rows[:n_calib]) * 1000
    yc = np.array(yaws[:n_calib]) * 1000
    pc = np.array(pitches[:n_calib]) * 1000

    lam_y = best_lambda(Xc, yc)
    lam_p = best_lambda(Xc, pc)
    by = ridge_fit(Xc, yc, lam_y)
    bp = ridge_fit(Xc, pc, lam_p)

    Xt = np.array(rows[n_calib:]) * 1000
    errs = []
    for i, x in enumerate(Xt):
        py = float(x @ by) / 1000
        pp = float(x @ bp) / 1000
        errs.append(angular_error(py, pp, gvecs[n_calib + i]))

    return {'mean': np.mean(errs), 'std': np.std(errs),
            'median': np.median(errs), 'n': len(errs)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mpii_root')
    ap.add_argument('--n-calib',  type=int, default=200)
    ap.add_argument('--subjects', type=int, default=15)
    ap.add_argument('--stride',   type=int, default=5,
                    help='Use every N-th frame after calibration set')
    args = ap.parse_args()

    orig_dir = Path(args.mpii_root) / 'Data' / 'Original'
    if not orig_dir.exists():
        sys.exit(f"ERROR: {orig_dir} not found")

    need_cnn = HAS_ONNX and os.path.exists(GAZE_ONNX)
    modes = ['pixel_sm', 'pixel_lg']
    if need_cnn:
        modes += ['cnn_only', 'hybrid', 'hybrid_lg']

    print(f"MPIIGaze ablation  |  n_calib={args.n_calib}  stride={args.stride}")
    print(f"CNN: {'enabled (' + GAZE_ONNX + ')' if need_cnn else 'disabled'}")
    print(f"Modes: {modes}\n")

    all_results = {m: [] for m in modes}
    n_done = 0

    for subj_dir in sorted(orig_dir.iterdir()):
        if not subj_dir.is_dir() or n_done >= args.subjects:
            break
        subj = subj_dir.name
        t0 = time.time()
        print(f"  {subj}: loading", end='', flush=True)

        samples = load_subject(subj_dir, args.n_calib, need_cnn, args.stride)
        print(f" {len(samples)} frames ({time.time()-t0:.0f}s) →", end=' ', flush=True)

        for mode in modes:
            r = run_mode(samples, mode, args.n_calib)
            if r:
                all_results[mode].append(r['mean'])
                print(f"{mode}={r['mean']:.1f}°", end='  ', flush=True)
        print()
        n_done += 1

    # Summary
    print(f"\n{'Mode':<12} {'Mean°':>8} {'±Std':>7}   description")
    print('─' * 60)
    descs = {
        'pixel_sm':  '10×6 hist-eq, 120-D (≈ Rust baseline)',
        'pixel_lg':  '20×12 hist-eq, 480-D',
        'cnn_only':  'MobileGaze (yaw,pitch) only, 2-D',
        'hybrid':    'pixel_sm + CNN, 122-D',
        'hybrid_lg': 'pixel_lg + CNN, 482-D',
    }
    for mode in modes:
        v = all_results[mode]
        if v:
            print(f"{mode:<12} {np.mean(v):>7.2f}° ±{np.std(v):.2f}°   {descs[mode]}")
        else:
            print(f"{mode:<12}  no data")

    print(f"\nLiterature: L2CS-Net 3.92° (no calib), FAZE 3.18° (9-pt calib)")


if __name__ == '__main__':
    main()
