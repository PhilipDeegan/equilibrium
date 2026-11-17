#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

PYTHONPATH=$PWD:$PWD/tpp/phlop python3 -m phlop.run.test_cases -i tst -p
PYTHONPATH=$PWD:$PWD/tpp/phlop python3 -m phlop.run.test_cases -i tst -c 2
