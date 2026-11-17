import re
import sys
from collections import defaultdict
from pathlib import Path

HEADER = re.compile(r"Header=(BB\d+_\d+)")
CHILD = re.compile(r"Child Loop (BB\d+_\d+)")
MARK = re.compile(r"#\s*LLVM-MCA-(BEGIN|END)\s+(\S+)")
SKIP = re.compile(r"^\s*(\.(loc|cfi_|p2align|file)|#APP|#NO_APP|#|\.Ltmp\d+:)")


def blocks(lines):
    cur, out = None, []
    for ln in lines:
        if ln.startswith("# %bb.") or re.match(r"(\.LBB\d+_\d+|[A-Za-z_][\w.$]*):", ln):
            cur = {"label": ln.split(":")[0].split()[-1], "head": ln, "body": []}
            out.append(cur)
        elif cur is not None:
            cur["body"].append(ln)
    return out


def loop_of(blk):
    h = HEADER.search(blk["head"])
    if h:
        return h.group(1)
    if "Loop Header" in blk["head"]:
        return blk["label"].lstrip(".L")
    return None


def code(sel):
    return [l for x in sel for l in x["body"] if l.strip() and not SKIP.match(l)]


def extract(asm):
    bl = blocks(asm.read_text(errors="replace").splitlines())
    loops = [loop_of(b) for b in bl]
    members, kids = defaultdict(list), {}
    for b, lp in zip(bl, loops):
        if lp:
            members[lp].append(b)
        if "Loop Header" in b["head"]:
            kids[lp] = [m.group(1) for l in b["body"] for m in [CHILD.search(l)] if m]

    regions = {}
    for i, b in enumerate(bl):
        for ln in b["body"]:
            m = MARK.search(ln)
            if not m or m.group(1) != "BEGIN" or m.group(2) in regions:
                continue
            name, loop = m.group(2), loops[i]
            if loop:
                members_order = members[loop] + [y for c in kids.get(loop, []) for y in members[c]]
                for c in kids.get(loop, []):
                    regions[f"{name}.{c}"] = (c, len(members[c]), code(members[c]))
                sel = members_order
            else:
                sel = []
                for x in bl[i:]:
                    sel.append(x)
                    if any(l.startswith(".Lfunc_end") for l in x["body"]):
                        break
                for x in sel:
                    if "Loop Header" in x["head"]:
                        lp = loop_of(x)
                        sub = members[lp] + [y for c in kids.get(lp, []) for y in members[c]]
                        regions[f"{name}.{lp}"] = (lp, len(sub), code(sub))
            regions[name] = (loop, len(sel), code(sel))
    return regions


if __name__ == "__main__":
    asm, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, (loop, nb, body) in extract(asm).items():
        (out / f"{name}.s").write_text("\n".join(body) + "\n")
        print(f"{name:34s} loop={loop} blocks={nb} instrs={len(body)}")
