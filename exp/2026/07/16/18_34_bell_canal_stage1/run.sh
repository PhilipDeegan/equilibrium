#!/usr/bin/env bash
# Bell/canal probe — stage 1 (separable baseline). Exact commands used.
set -e
cd "$(git rev-parse --show-toplevel)"

# quick baseline (foreground validation run)
PYTHONPATH=$PWD python3 -u tst/substrate/probe_bell_canal.py \
  --checkpoints 0,25,50,100,200 --samples 200000

# long horizon, noise-free (through and past the A->B wave-crossing time)
PYTHONPATH=$PWD python3 -u tst/substrate/probe_bell_canal.py \
  --checkpoints 0,500,1000,2000,4000,8000 --samples 200000 \
  --outdir bell_long

# bulk ambient-noise decoherence attempt (eta = 0.01, 16 realizations)
PYTHONPATH=$PWD python3 -u tst/substrate/probe_bell_canal.py \
  --checkpoints 0,500,1000,2000,4000 --samples 200000 \
  --noise 0.01 --noise-mode bulk --noise-k 16 \
  --outdir bell_noise
