import sys

from . import log, math, physical, plot, setup, substrate

__all__ = ["log", "math", "physical", "plot", "setup", "substrate"]


def lz(ar, s=0, **kwargs):
    from equilibrium import cpp as eq_cpp  # pylint: disable = no-name-in-module

    if isinstance(ar, eq_cpp.lazy_np_arr_d):
        return eq_cpp.lazy_eval_d(ar)

    return eq_cpp.lazy_np_arr_d(eq_cpp.make_aligned_size_d(ar, s))


def loaded_modules_in(all, pre="equilibrium"):
    mods = []
    for m in all:
        mod = f"{pre}.{m}"
        if mod in sys.modules:
            mods += [sys.modules[mod]]
            if hasattr(sys.modules[mod], "__all__"):
                mods += loaded_modules_in(sys.modules[mod].__all__, f"{mod}")
    return mods


def import_lib_reload():
    """
    force jupyter to update modules and take latest code
    """

    import importlib

    importlib.reload(sys.modules[__name__])
    for lm in reversed(loaded_modules_in(__all__)):
        importlib.reload(lm)
