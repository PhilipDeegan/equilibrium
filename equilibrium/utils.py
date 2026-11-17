from pathlib import Path

import pydantic


def ensure_dir(path):
    """Create the parent directory of `path` (or `path` itself if it looks
    like a directory, i.e. has no suffix) if it doesn't already exist.
    No-op if `path` is falsy (None/""), so callers with an optional
    --out-dir-style arg can call this unconditionally.

    Call this up front, before expensive work, for any output path a script
    plans to write to later -- catching a bad/missing directory immediately
    is much cheaper than discovering it via a FileNotFoundError from
    plt.savefig() after hours of computation.
    """
    if not path:
        return None
    p = Path(path)
    d = p if p.suffix == "" else p.parent
    d.mkdir(parents=True, exist_ok=True)
    return p


def as_dict(obj):
    return {key: getattr(obj, key) for key in obj.__dict__.keys()}


def to_string(obj, s=""):
    for k, v in as_dict(obj).items():
        s += k + ", " + str(v) + "\n"
    return s[:-1]


def as_yaml(obj):
    import yaml

    return yaml.dump(obj)


def raise_if(b, s="Fail"):
    if b:
        raise ValueError(s)


class T(pydantic.BaseModel):
    n: str
    v: float = 0
    # v :
    add: list["T"] = []
    sub: list["T"] = []
    mul: list["T"] = []
    div: list["T"] = []

    def calc(self):
        v = self.v
        v += sum([t.v for t in self.add])
        v -= sum([t.v for t in self.sub])
        for t in self.mul:
            v *= t.v
        for t in self.div:
            v /= t.v
        return T(n="result", v=v)

    def __call__(self, *args):
        if not args:
            return self.calc()
        assert len(args) == 1
        self.v = args[0]
        return self

    def __add__(self, t):
        self.add += [t]
        return self

    def __sub__(self, t):
        self.sub += [t]
        return self

    def __mul__(self, t):
        self.mul += [t]
        return self

    def __div__(self, t):
        self.div += [t]
        return self

    def __repr__(self):
        return f"{self.n}:{self.v}"
