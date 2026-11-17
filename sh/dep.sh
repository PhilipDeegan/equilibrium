#!/usr/bin/env bash
set -x
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
ROOT="$PWD"

fn_github_get(){
    git clone "https://github.com/${1}" --recurse-submodules --shallow-submodules --depth 4 ${*:2}
}

fn_sdl2_setup(){

    mkdir -p tpp
    (
        cd tpp

        # SDL2
        [ ! -d "SDL" ] && fn_github_get libsdl-org/SDL -b release-2.30.6 SDL
        (
            cd SDL
            [ ! -d "build" ] && mkdir build && cd build && \
            cmake .. -G Ninja -DCMAKE_C_FLAGS="-march=native" -DCMAKE_BUILD_TYPE=Release \
                     -DCMAKE_INSTALL_PREFIX="$ROOT/tpp/bin" && ninja && ninja install
            echo "SDL2 built"
        )

        # SDL_ttf
        [ ! -d "SDL_ttf" ] && fn_github_get libsdl-org/SDL_ttf -b release-2.22.0 SDL_ttf
        (
            cd SDL_ttf
            [ ! -d "build" ] && mkdir build && cd build && \
            cmake .. -G Ninja -DCMAKE_C_FLAGS="-march=native" -DCMAKE_BUILD_TYPE=Release \
                     -DSDL2_INCLUDE_DIR="$ROOT/tpp/bin/include" \
                     -DCMAKE_INSTALL_PREFIX="$ROOT/tpp/bin" && ninja && ninja install && \
            echo "SDL_ttf built"
        )

        # SDL_image
        [ ! -d "SDL_image" ] && fn_github_get libsdl-org/SDL_image -b release-2.8.2 SDL_image
        (
            cd SDL_image
            [ ! -d "build" ] && mkdir build && cd build && \
            cmake .. -G Ninja -DCMAKE_C_FLAGS="-march=native" -DCMAKE_BUILD_TYPE=Release \
                     -DSDL2_INCLUDE_DIR="$ROOT/tpp/bin/include" \
                     -DCMAKE_INSTALL_PREFIX="$ROOT/tpp/bin" && ninja && ninja install && \
            echo "SDL_image built"
        )
    )
}

fn_sdl2_dbg_setup(){

    mkdir -p tpp
    (
        cd tpp

        # SDL2
        [ ! -d "SDL" ] && fn_github_get libsdl-org/SDL -b release-2.30.6 SDL
        (
            cd SDL && rm -rf build
            [ ! -d "build" ] && mkdir build && cd build && \
            cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug \
                     -DCMAKE_INSTALL_PREFIX="$ROOT/tpp/dbg" && ninja && ninja install
            echo "SDL2 built"
        )

        # SDL_ttf
        [ ! -d "SDL_ttf" ] && fn_github_get libsdl-org/SDL_ttf -b release-2.22.0 SDL_ttf
        (
            cd SDL_ttf && rm -rf build
            [ ! -d "build" ] && mkdir build && cd build && \
            cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug \
                     -DSDL2_INCLUDE_DIR="$ROOT/tpp/dbg/include" \
                     -DCMAKE_INSTALL_PREFIX="$ROOT/tpp/dbg" && ninja && ninja install && \
            echo "SDL_ttf built"
        )

        # SDL_image
        [ ! -d "SDL_image" ] && fn_github_get libsdl-org/SDL_image -b release-2.8.2 SDL_image
        (
            cd SDL_image && rm -rf build
            [ ! -d "build" ] && mkdir build && cd build && \
            cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug \
                     -DSDL2_INCLUDE_DIR="$ROOT/tpp/dbg/include" \
                     -DCMAKE_INSTALL_PREFIX="$ROOT/tpp/dbg" && ninja && ninja install && \
            echo "SDL_image built"
        )
    )
}

I=$(echo $1 | tr -d '\r')
[ "$I" = "sdl" ] && fn_sdl2_setup
[ "$I" = "sdl_dbg" ] && fn_sdl2_dbg_setup
