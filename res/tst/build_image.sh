#!/usr/bin/env bash
set -ex
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd $CWD

FULL_NAME="debian:sdl2"
docker build -t ${FULL_NAME} .
