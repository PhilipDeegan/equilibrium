#!/usr/bin/env bash
set -ex

cd /root

apt update && apt install -y bash-completion cmake git flex bison gawk m4 make ninja-build python3-dev sudo g++ \
                            libx11-dev libxext-dev libfreetype-dev libssl-dev gedit python3-pip python3-venv

git clone https://github.com/mkn/mkn --recurse-submodules --shallow-submodules --depth 10 /root/mkn
git clone https://github.com/libsdl-org/SDL -b release-2.30.6 SDL2 --recurse-submodules --shallow-submodules --depth 10
git clone https://github.com/libsdl-org/SDL_ttf -b release-2.22.0  SDL2_ttf --depth 10 --recurse-submodules --shallow-submodules

(
    cd SDL2 && mkdir -p build && cd build
    cmake .. -G Ninja -DCMAKE_C_FLAGS="-march=native" -DCMAKE_BUILD_TYPE=Release
    ninja && ninja install
)
(
    cd SDL2_ttf && mkdir -p build && cd build
    cmake .. -G Ninja -DCMAKE_C_FLAGS="-march=native" -DCMAKE_BUILD_TYPE=Release #\
             #DSDL2_LIBRARY=/home/p/SDL2/build/libSDL2-2.0.so -DSDL2_INCLUDE_DIR=/home/p/SDL2/include
    ninja && ninja install
)

rm -rf SDL2 SDL2_ttf

(
    cd mkn && ./res/ci/nixish_setup.sh && make nix
    ./mkn clean build -dtKOw mkn.ram[https]
    cp bin/build/mkn /usr/local/bin/mkn
)
