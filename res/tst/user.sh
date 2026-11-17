#!/usr/bin/env bash
set -ex

cd ~/
whoami
git clone https://github.com/carltheperson/blockamok --depth 4

(
    cd ~/blockamok
    [ ! -f "mkn.yaml" ] && cat >mkn.yaml <<EOL
name: blockamok
inc: /usr/local/include/SDL2
lib: SDL2-2.0 SDL2_ttf-2.0
path: /usr/local/lib
src: src
main: src/main.c
EOL

    mkn clean build -Ol "-lm -lfreetype"
)
