#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -e

PYTHONPATH=$PWD python3 -u vis/dedupe_images.py "$@"
