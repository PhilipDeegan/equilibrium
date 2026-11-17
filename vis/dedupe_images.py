"""Flag near-duplicate images in a directory by windowed SSIM (Wang et al.
2004), so near-identical plot frames can be deduplicated by "how much
changed" rather than a binary file-hash comparison. No skimage/imagehash
dependency -- SSIM implemented directly with scipy.ndimage gaussian filters
for the local mean/variance/covariance terms.
"""

import argparse
import glob
import os
import re

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def ssim_map(a, b, sigma=1.5):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    var_a = gaussian_filter(a * a, sigma) - mu_a2
    var_b = gaussian_filter(b * b, sigma) - mu_b2
    cov_ab = gaussian_filter(a * b, sigma) - mu_ab

    num = (2 * mu_ab + C1) * (2 * cov_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (var_a + var_b + C2)
    return num / den


def ssim(a, b, sigma=1.5, mask=None):
    smap = ssim_map(a, b, sigma)
    return float(smap[mask].mean()) if mask is not None else float(smap.mean())


def natural_key(path):
    m = re.search(r"(\d+)(?!.*\d)", os.path.basename(path))
    return int(m.group(1)) if m else os.path.basename(path)


def auto_roi_mask(imgs, quantile, pad):
    """Mask of the pixels that actually vary across the stack, so static
    chrome (axes, titles, colorbars, background) can't dilute the SSIM
    average -- a bounding-box crop still includes any static pixels inside
    the box, a mask excludes them directly."""
    stack = np.stack(imgs)
    std = stack.std(axis=0)
    mask = std > np.quantile(std, quantile)
    if pad > 0:
        mask = binary_dilation(mask, iterations=pad)
    return mask


def main():
    ap = argparse.ArgumentParser(
        description="Flag near-duplicate images by windowed SSIM."
    )
    ap.add_argument("dir", help="directory of images to scan")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.995,
        help="SSIM above this counts as near-duplicate",
    )
    ap.add_argument(
        "--sigma", type=float, default=1.5, help="gaussian window sigma for local stats"
    )
    ap.add_argument(
        "--against",
        choices=["prev", "first"],
        default="prev",
        help="compare each frame to the previous frame (drift per step), or all frames to the first (cumulative drift)",
    )
    ap.add_argument(
        "--list-duplicates",
        action="store_true",
        help="print only the path of each flagged near-duplicate, one per line (for piping to xargs)",
    )
    ap.add_argument(
        "--auto-roi",
        action="store_true",
        help="average SSIM only over pixels that actually vary across the stack, "
        "so static chrome (axes/titles/colorbars/background) can't dilute the score",
    )
    ap.add_argument(
        "--roi-quantile",
        type=float,
        default=0.9,
        help="--auto-roi: std-dev quantile that counts as 'varies'",
    )
    ap.add_argument(
        "--roi-pad",
        type=int,
        default=3,
        help="--auto-roi: binary-dilation iterations to grow the mask",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)), key=natural_key)
    if len(paths) < 2:
        print(
            f"need >= 2 images matching {args.glob} in {args.dir}, found {len(paths)}"
        )
        return

    imgs = [load_gray(p) for p in paths]

    mask = None
    if args.auto_roi:
        mask = auto_roi_mask(imgs, args.roi_quantile, args.roi_pad)
        print(
            f"auto-roi: {mask.sum()} / {mask.size} pixels ({100 * mask.sum() / mask.size:.1f}%)"
        )

    scores = []
    for i in range(1, len(imgs)):
        b_idx = i - 1 if args.against == "prev" else 0
        s = ssim(imgs[b_idx], imgs[i], args.sigma, mask=mask)
        scores.append(s)
        is_dup = s > args.threshold
        if args.list_duplicates:
            if is_dup:
                print(paths[i])
            continue
        flag = "  <- near-duplicate" if is_dup else ""
        print(
            f"{os.path.basename(paths[b_idx]):>24} {os.path.basename(paths[i]):>24} {s:8.5f}{flag}"
        )

    if not args.list_duplicates:
        scores = np.array(scores)
        n_dup = int((scores > args.threshold).sum())
        print(f"\n{len(paths)} images, {len(scores)} pairs (against={args.against})")
        print(
            f"SSIM: min={scores.min():.5f} max={scores.max():.5f} mean={scores.mean():.5f}"
        )
        print(f"near-duplicates (SSIM > {args.threshold}): {n_dup} / {len(scores)}")


if __name__ == "__main__":
    main()
