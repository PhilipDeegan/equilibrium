#!/usr/bin/env bash
# Spheromak decay parameter sweep — 200k steps per permutation (4 × 50k chunks).
# Shared IC per lattice type (ring IC is cross/lam-scale independent).
# probe_decay_compare runs at end of each permutation — per-chunk plots suppressed.
# Skips any chunk whose .npz already exists, so reruns are safe.

set -eo pipefail

SWEEP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( cd "$SWEEP_DIR/../../../.." && pwd )"

export PYTHONPATH="$REPO"
cd "$REPO"

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
CROSS_SCALES=("0.001" "0.005" "0.010")
LAM_SCALES=("0.020" "0.050" "0.100")
LATTICES=("uniform" "adaptive")

CHUNK=50000       # steps per chunk  →  4 × 50k = 200k total
N_CHUNKS=4
REPORT=5000
R=40
N_THREADS=10      # adaptive lattice worker threads
# ---------------------------------------------------------------------------

_run_uniform() {
    # usage: _run_uniform <load|""> <save> <cs> <ls> <n>
    local load="$1" save="$2" cs="$3" ls="$4" n="$5"
    local args=()
    [[ -n "$load" ]] && args+=("--load" "$load")
    python3 tst/substrate/test_spheromak_decay.py \
        --R "$R" --rings both \
        "${args[@]}" \
        --cross-scale "$cs" --lam-scale "$ls" \
        --n-steps "$n" --report "$REPORT" \
        --save-snapshot "$save" \
        --no-plot
}

_run_adaptive() {
    # usage: _run_adaptive <load|""> <save> <cs> <ls> <n>
    local load="$1" save="$2" cs="$3" ls="$4" n="$5"
    local args=()
    [[ -n "$load" ]] && args+=("--load" "$load")
    python3 tst/substrate/adaptive/test_spheromak_decay.py \
        --R "$R" --rings both \
        "${args[@]}" \
        --cross-scale "$cs" --lam-scale "$ls" \
        --n-steps "$n" --report "$REPORT" \
        --n-threads "$N_THREADS" \
        --save-snapshot "$save" \
        --no-plot
}

_compare_uniform() {
    # usage: _compare_uniform <plot-dir> <npz files...>
    local pdir="$1"; shift
    python3 tst/substrate/probe_decay_compare.py \
        --R "$R" --plot-dir "$pdir" "$@"
    python3 tst/substrate/probe_decay_compare.py \
        --R "$R" --plot-dir "$pdir" --zoom 10 "$@"
}

_compare_adaptive() {
    # usage: _compare_adaptive <plot-dir> <npz files...>
    local pdir="$1"; shift
    python3 tst/substrate/adaptive/probe_decay_compare.py \
        --plot-dir "$pdir" "$@"
    python3 tst/substrate/adaptive/probe_decay_compare.py \
        --plot-dir "$pdir" --zoom 10 "$@"
}

# ---------------------------------------------------------------------------
# Shared ICs (independent of cross/lam scales — only ring geometry matters)
# ---------------------------------------------------------------------------
for lat in "${LATTICES[@]}"; do
    ic_npz="$SWEEP_DIR/${lat}_ic.npz"
    if [[ -f "$ic_npz" ]]; then
        echo "=== IC exists, skipping: $lat ==="
        continue
    fi
    echo "=== Generating IC: $lat ===" && date
    if [[ "$lat" == "uniform" ]]; then
        _run_uniform "" "$ic_npz" "0.005" "0.05" 0
    else
        _run_adaptive "" "$ic_npz" "0.005" "0.05" 0
    fi
done

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for lat in "${LATTICES[@]}"; do
    ic_npz="$SWEEP_DIR/${lat}_ic.npz"
    for cs in "${CROSS_SCALES[@]}"; do
        for ls in "${LAM_SCALES[@]}"; do
            tag="${lat}_cs${cs}_ls${ls}"
            pdir="$SWEEP_DIR/$tag"
            mkdir -p "$pdir"

            echo ""
            echo "=== $tag ===" && date

            prev="$ic_npz"
            for chunk in $(seq 1 "$N_CHUNKS"); do
                snap="$pdir/stage${chunk}.npz"
                log="$pdir/stage${chunk}.log"

                if [[ -f "$snap" ]]; then
                    echo "  chunk $chunk: exists, skipping"
                    prev="$snap"
                    continue
                fi

                start_step=$(( (chunk - 1) * CHUNK ))
                end_step=$(( chunk * CHUNK ))
                echo "  chunk $chunk / $N_CHUNKS  (steps ${start_step}–${end_step})"

                if [[ "$lat" == "uniform" ]]; then
                    _run_uniform "$prev" "$snap" "$cs" "$ls" "$CHUNK" 2>&1 | tee "$log"
                else
                    _run_adaptive "$prev" "$snap" "$cs" "$ls" "$CHUNK" 2>&1 | tee "$log"
                fi
                prev="$snap"
            done

            # probe_decay_compare: IC + all completed stages
            snaps=("$ic_npz")
            for chunk in $(seq 1 "$N_CHUNKS"); do
                [[ -f "$pdir/stage${chunk}.npz" ]] && snaps+=("$pdir/stage${chunk}.npz")
            done
            echo "  probe_decay_compare (${#snaps[@]} snapshots)" && date
            if [[ "$lat" == "uniform" ]]; then
                _compare_uniform "$pdir" "${snaps[@]}" 2>&1 | tee "$pdir/compare.log"
            else
                _compare_adaptive "$pdir" "${snaps[@]}" 2>&1 | tee "$pdir/compare.log"
            fi

            echo "  done: $tag" && date
        done
    done
done

echo ""
echo "=== Sweep complete ===" && date
