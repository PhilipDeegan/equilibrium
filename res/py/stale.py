#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def parse_size(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?", s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s}")
    return int(float(m.group(1)) * UNITS[m.group(2)])


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def tracked_files(root):
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        out = subprocess.run(
            ["git", "-C", top, "ls-files", "-z", "--", str(root.resolve())],
            capture_output=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return {(Path(top) / p).resolve() for p in out.decode().split("\0") if p}


def scan(root, min_size, max_mtime, exts):
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            if exts and p.suffix.lstrip(".").lower() not in exts:
                continue
            try:
                st = p.lstat()
            except OSError:
                continue
            if st.st_size >= min_size and st.st_mtime <= max_mtime:
                yield p, st.st_size, st.st_mtime


def main():
    ap = argparse.ArgumentParser(
        description="List large and/or old files under a path (read-only)."
    )
    ap.add_argument("path", type=Path)
    ap.add_argument("-s", "--min-size", type=parse_size, default=0,
                    help="e.g. 50M, 1G (default 0)")
    ap.add_argument("-d", "--days", type=float, default=0,
                    help="only files not modified in this many days")
    ap.add_argument("-e", "--ext", action="append", default=[],
                    help="restrict to extension(s), repeatable: -e npz -e npy")
    ap.add_argument("-g", "--group", type=int, metavar="DEPTH",
                    help="aggregate per directory at DEPTH below path")
    ap.add_argument("-n", "--top", type=int, default=50,
                    help="rows to show (0 = all)")
    ap.add_argument("--sort", choices=("size", "age"), default="size")
    ap.add_argument("--untracked", action="store_true",
                    help="skip files tracked by git")
    ap.add_argument("-p", "--paths", action="store_true",
                    help="print only paths, one per line")
    ap.add_argument("-0", "--print0", action="store_true",
                    help="print only paths, NUL-separated (for xargs -0)")
    a = ap.parse_args()

    if not a.path.is_dir():
        sys.exit(f"not a directory: {a.path}")

    now = time.time()
    exts = {e.lstrip(".").lower() for e in a.ext}
    rows = list(scan(a.path, a.min_size, now - a.days * 86400, exts))

    tracked = tracked_files(a.path)
    if a.untracked and tracked is not None:
        rows = [r for r in rows if r[0].resolve() not in tracked]

    if a.paths or a.print0:
        rows.sort(key=(lambda r: -r[1]) if a.sort == "size" else (lambda r: r[2]))
        end = "\0" if a.print0 else "\n"
        for p, _, _ in rows:
            sys.stdout.write(f"{p}{end}")
        return

    total = sum(r[1] for r in rows)

    if a.group is not None:
        agg = defaultdict(lambda: [0, 0, 0.0])
        for p, size, mtime in rows:
            parts = p.relative_to(a.path).parts[:-1][: a.group]
            g = agg[Path(a.path, *parts)]
            g[0] += size
            g[1] += 1
            g[2] = max(g[2], mtime)
        items = sorted(agg.items(),
                       key=(lambda kv: -kv[1][0]) if a.sort == "size"
                       else (lambda kv: kv[1][2]))
        if a.top:
            items = items[: a.top]
        print(f"{'size':>8} {'files':>6} {'newest':>10}  dir")
        for d, (size, n, mtime) in items:
            print(f"{human(size):>8} {n:>6} "
                  f"{time.strftime('%Y-%m-%d', time.localtime(mtime))}  {d}")
    else:
        rows.sort(key=(lambda r: -r[1]) if a.sort == "size" else (lambda r: r[2]))
        shown = rows[: a.top] if a.top else rows
        print(f"{'size':>8} {'modified':>10} {'git':>3}  path")
        for p, size, mtime in shown:
            t = "?" if tracked is None else ("T" if p.resolve() in tracked else "")
            print(f"{human(size):>8} "
                  f"{time.strftime('%Y-%m-%d', time.localtime(mtime))} {t:>3}  {p}")

    print(f"\n{len(rows)} files, {human(total)}", file=sys.stderr)


if __name__ == "__main__":
    main()
