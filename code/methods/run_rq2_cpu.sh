#!/usr/bin/env bash
# RQ2 open-set runs that are CPU-bound, so they can run alongside run_rq2_gpu.sh. Resumable: a method whose open_set_predictions.csv
# already exists is skipped.
#
#   cd code/methods && bash run_rq2_cpu.sh
#
# EDIT THE FOUR PATHS BELOW. The first points inside this package; the other three point
# at data the package does not ship (see README.md, "What is included and what is not").
set -uo pipefail

PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # the submission/ root
PY=${PY:-python}                                          # your interpreter
IMGFLIP_ROOT=/path/to/imgflip-canonical                   # the ImgFlip image tree
SOCIAL_ROOT=/path/to/social-media                         # the unlabelled discovery corpus
NONMEME_ROOT=/path/to/non-memes                           # the non-meme corpus

TRAIN=$PKG/data/splits/imgflip_80_20_seed40/train.parquet
TEST=$PKG/data/splits/imgflip_80_20_seed40/test.parquet
SAMPLE=$PKG/data/rq2_benchmark/images
SOCIAL="$SOCIAL_ROOT/Facebook2023_nolabel,$SOCIAL_ROOT/Reddit2024_nolabel/images,$SOCIAL_ROOT/Twitter2023_nolabel"
OUT=$PKG/results/rq2
LOGDIR=$PKG/results/logs
NONMEME="--non-meme-root $NONMEME_ROOT --max-non-meme-images 20000"
REMAP="--path-prefix-from /path/to/imgflip-canonical --path-prefix-to $IMGFLIP_ROOT"
mkdir -p "$OUT" "$LOGDIR"
export PYTHONWARNINGS="ignore::UserWarning"
cd "$PKG/code/methods"

done_already() { find "$OUT/$1" -name open_set_predictions.csv 2>/dev/null | grep -q .; }

run() {
  local name=$1; shift
  if done_already "$name"; then echo "[skip] $name already has open_set_predictions.csv"; return; fi
  echo; echo "=================== $name   $(date '+%F %T') ==================="
  "$@" 2>&1 | tee "$LOGDIR/open_set_$name.log"
  echo "=================== $name finished $(date '+%F %T') ==================="
}

# ~15 min
run rnn_phash $PY meme_research_rnn_phash_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 --val-size 0.10 \
  --radius-values 6,8,10,12,14,16,20,24,32 --weights uniform,distance \
  $NONMEME --open-set-dir $SAMPLE --output-dir $OUT/rnn_phash

# Long. The 30k-image ImgFlip test pass is the cost; --max-test-images 200 shrinks it so
# the 1,000 open-set predictions are what you wait for. RQ1 numbers already exist elsewhere.
run sparse_matching $PY meme_research_sparse_matching_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 \
  --max-refs-per-template 30 --target-size 64 --alpha 0.35 --max-iter 1000 --num-workers 6 \
  --max-test-images 200 $NONMEME \
  --open-set-dir $SAMPLE --output-dir $OUT/sparse_matching

# Longest. d=27, m=20 as in RQ1; add 4,10 to --min-match-values later for the reviewer sweep.
run rnn_feature_matching $PY meme_research_rnn_feature_matching_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 --val-size 0.10 \
  --max-refs-per-template 20 --target-size 256 --n-keypoints 512 --fast-threshold 0.08 \
  --radius-values 0.90,0.93,0.95,0.97,0.99 --weights uniform,distance --num-workers 6 \
  --max-test-images 200 $NONMEME \
  --open-set-dir $SAMPLE --output-dir $OUT/rnn_feature_matching

echo; echo "CPU chain complete $(date '+%F %T')"
