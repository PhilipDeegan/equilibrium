"""move flat <name>_tNNNNN.png / <name>[_fromT]_rec.npz into <name split on _>/tNNNNN.png, rec[_fromT].npz."""
import re
import sys
from pathlib import Path

pat = re.compile(r"^(?P<name>.+?)_(?:(?P<tag>t\d+)\.png|(?:from(?P<frm>\d+)_)?rec\.npz)$")
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
for d in sys.argv[1:2]:
    d = Path(d)
    for f in sorted(d.iterdir()):
        mt = pat.match(f.name)
        if not f.is_file() or not mt or not f.name.startswith(prefix):
            continue
        out = d / Path(*mt["name"].split("_"))
        out.mkdir(parents=True, exist_ok=True)
        new = f"{mt['tag']}.png" if mt["tag"] else (f"rec_from{mt['frm']}.npz" if mt["frm"] else "rec.npz")
        f.rename(out / new)
