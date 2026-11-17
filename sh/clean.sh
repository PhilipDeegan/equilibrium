#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/.. && CWD=$PWD


RM_RF=(
  __pycache__
  .ipynb_checkpoints
)

(
  for RM in "${RM_RF[@]}"; do
    find . -name "${RM}" | xargs rm -rf
  done
)
