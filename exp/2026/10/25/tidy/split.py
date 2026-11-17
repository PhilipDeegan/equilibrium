import re
import sys
from pathlib import Path

root = Path("/home/p/git/eq")
dry = "--dry" in sys.argv


def split(name):
    text = (root / name).read_text()
    parts = re.split(r"(?m)^(?=## )", text)
    out, date = {}, None
    for p in parts:
        if not p.strip():
            continue
        m = re.search(r"2026-(\d\d)-(\d\d)", p.splitlines()[0])
        if m:
            date = m.groups()
        if date is None:
            date = ("10", "24")
        out.setdefault(date, []).append(p.rstrip() + "\n")
    for (mm, dd), secs in out.items():
        d = root / "exp/2026" / mm / dd
        f = d / f"{name}.md"
        print(f, len(secs), sum(len(s) for s in secs))
        if dry:
            continue
        d.mkdir(parents=True, exist_ok=True)
        head = f"# Entries for 2026-{mm}-{dd} moved from repo-root `{name}` on 2026-10-25 (verbatim, newest first)\n\n"
        f.write_text(head + "\n".join(secs))


for n in sys.argv[1:]:
    if n != "--dry":
        split(n)
