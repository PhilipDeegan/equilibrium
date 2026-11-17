import os
import subprocess
import sys


def install(*args):
    pip_install = 0
    if "EQ_FORCE_PIP_INSTALL" in os.environ:
        print("EQ: envvar EQ_FORCE_PIP_INSTALL detected")
        pip_install = os.environ.get("EQ_FORCE_PIP_INSTALL", 0)
    elif not pip_install:
        pip_install = input("enter 1 to do a pip install")
    if pip_install:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + list(args))


if __name__ != "__main__":
    import importlib

    find = importlib.util.find_spec("pydantic")
    if not find:
        print("EQ: pydantic not found - pip install requirements.txt ?")
        install("-r", "requirements.txt")
