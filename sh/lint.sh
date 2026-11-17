#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -ex
(
  cd "$CWD"/..
  export PYTHONPATH=$PWD

  PY_FILES=$(find equilibrium tst vis -name "*.py")
  python3 -m black equilibrium tst vis exp run
  python3 -m ruff check equilibrium tst vis exp run
  # pylint --load-plugins pylint_pydantic --extension-pkg-whitelist='pydantic' --errors-only equilibrium tst vis
  # isort equilibrium tst
  for FILE in ${PY_FILES[@]}; do
    autoflake -i "$FILE"
  done
) 1> >(tee $CWD/.lint.sh.out ) 2> >(tee $CWD/.lint.sh.err >&2 )
