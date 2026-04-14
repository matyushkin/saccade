#!/usr/bin/env python3
"""
Preprocess MPIIGaze dataset: convert .mat files to PNG images + TSV labels.

MPIIGaze Normalized data (Data/Normalized/p*/day*.mat) contains:
  data.left.image   (N, 36, 60)  uint8 left eye patches
  data.right.image  (N, 36, 60)  uint8 right eye patches
  data.left.gaze    (N, 3)       3-D unit gaze vector in normalized camera frame
  data.left.pose    (N, 3)       head pose

Gaze vector convention (MPIIGaze normalized space):
  x = right, y = down, z = toward camera
  theta = asin(-y)        → pitch  (positive = looking up)
  phi   = atan2(-x, -z)   → yaw   (positive = looking left from subject POV)

Usage:
  python3 tools/preprocess_mpii.py ./MPIIGaze ./MPIIGaze_proc

Output:
  MPIIGaze_proc/p00/
    labels.tsv   columns: idx  gx  gy  gz  head0  head1  head2
    0000000_left.png   (36x60 grayscale)
    0000000_right.png
"""

import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, trying h5py")


def load_mat(path):
    if HAS_SCIPY:
        try:
            return sio.loadmat(str(path))
        except Exception:
            pass
    import h5py
    def h5_to_dict(group):
        result = {}
        for k in group:
            v = group[k]
            result[k] = h5_to_dict(v) if hasattr(v, 'keys') else v[()]
        return result
    with h5py.File(str(path), 'r') as f:
        return h5_to_dict(f)


def extract_data(mat):
    """Return (left_imgs, right_imgs, gaze_3d, pose) all as numpy arrays."""
    d = mat['data']

    # scipy structured array layout
    if hasattr(d, 'dtype') and d.dtype.names:
        left  = d['left'][0, 0]
        right = d['right'][0, 0]
        l_img   = left['image'][0, 0]    # (N,36,60)
        r_img   = right['image'][0, 0]
        l_gaze  = left['gaze'][0, 0]     # (N,3) unit vector
        l_pose  = left['pose'][0, 0]     # (N,3)
    else:
        l_img   = d['left']['image']
        r_img   = d['right']['image']
        l_gaze  = d['left']['gaze']
        l_pose  = d['left']['pose']

    l_img  = np.asarray(l_img,  dtype=np.uint8)
    r_img  = np.asarray(r_img,  dtype=np.uint8)
    l_gaze = np.asarray(l_gaze, dtype=np.float64)
    l_pose = np.asarray(l_pose, dtype=np.float32)

    # Ensure (N, H, W) — MATLAB stores in column-major so might be (H, W, N)
    if l_img.ndim == 3 and l_img.shape[0] < l_img.shape[2]:
        # (H, W, N) → (N, H, W)
        l_img = l_img.transpose(2, 0, 1)
        r_img = r_img.transpose(2, 0, 1)

    # Ensure (N, 3) for gaze/pose
    if l_gaze.ndim == 2 and l_gaze.shape[0] == 3 and l_gaze.shape[1] != 3:
        l_gaze = l_gaze.T
    if l_pose.ndim == 2 and l_pose.shape[0] == 3 and l_pose.shape[1] != 3:
        l_pose = l_pose.T

    return l_img, r_img, l_gaze, l_pose


def process_subject(subject_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / "labels.tsv"

    total = 0
    with open(label_path, 'w') as lf:
        lf.write("idx\tgx\tgy\tgz\thead0\thead1\thead2\n")

        for mat_path in sorted(subject_dir.glob("day*.mat")):
            try:
                mat = load_mat(mat_path)
                l_img, r_img, gaze, pose = extract_data(mat)
            except Exception as e:
                print(f"  WARNING: skip {mat_path.name}: {e}", file=sys.stderr)
                continue

            N = gaze.shape[0]
            for i in range(N):
                idx = total + i
                gx, gy, gz = float(gaze[i, 0]), float(gaze[i, 1]), float(gaze[i, 2])
                h0, h1, h2 = float(pose[i, 0]), float(pose[i, 1]), float(pose[i, 2])

                Image.fromarray(l_img[i]).save(str(out_dir / f"{idx:07}_left.png"))
                Image.fromarray(r_img[i]).save(str(out_dir / f"{idx:07}_right.png"))
                lf.write(f"{idx}\t{gx:.6f}\t{gy:.6f}\t{gz:.6f}\t{h0:.4f}\t{h1:.4f}\t{h2:.4f}\n")

            total += N
            print(f"  {mat_path.name}: {N} samples (total {total})")

    return total


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <MPIIGaze_root> <output_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    out  = Path(sys.argv[2])
    norm_dir = root / "Data" / "Normalized"

    if not norm_dir.exists():
        print(f"ERROR: {norm_dir} not found.")
        sys.exit(1)

    print(f"MPIIGaze preprocessor: {norm_dir} → {out}")
    grand_total = 0

    for subject_dir in sorted(norm_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        print(f"\n{subject_dir.name}:")
        n = process_subject(subject_dir, out / subject_dir.name)
        grand_total += n

    print(f"\nDone. Total samples: {grand_total}")
    print(f"Output: {out}")
    print(f"\nRun benchmark: cargo run --release --example mpii_bench -- {out}")


if __name__ == "__main__":
    main()
