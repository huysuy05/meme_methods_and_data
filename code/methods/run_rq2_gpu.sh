#!/usr/bin/env bash
# RQ2 open-set runs that need a GPU. Resumable: a method whose open_set_predictions.csv
# already exists is skipped.
#
#   cd code/methods && bash run_rq2_gpu.sh
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

run() {  # run <name> <command...>
  local name=$1; shift
  if done_already "$name"; then echo "[skip] $name already has open_set_predictions.csv"; return; fi
  echo; echo "=================== $name   $(date '+%F %T') ==================="
  "$@" 2>&1 | tee "$LOGDIR/open_set_$name.log"
  echo "=================== $name finished $(date '+%F %T') ==================="
}

# ~3 min: reloads its own two checkpoints; stage 1 already supplies Non-Meme, so no --non-meme-root
run two_stage_densenet121 $PY meme_research_two_stage_cnn_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 \
  --eval-batch-size 64 --num-workers 8 \
  --checkpoint-dir $REPO/results/runs/meme_research_two_stage_cnn_eval/20260830_182011_batch/seed_40 \
  --open-set-dir $SAMPLE --output-dir $OUT/two_stage_densenet121

# ~15 min: embeds with the seed-40 DenseNet trunk
run rnn_densenet_embedding $PY meme_research_rnn_densenet_embedding_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 --val-size 0.10 \
  --batch-size 64 --num-workers 12 \
  --radius-values 0.1,0.15,0.18,0.2,0.22,0.25,0.3 --weights uniform,distance \
  --cnn-checkpoint $REPO/results/runs/meme_research_cnn_eval/densenet121/20260818_170124_batch/seed_40/best_model.pt \
  $NONMEME --open-set-dir $SAMPLE --output-dir $OUT/rnn_densenet_embedding

# ~25 min
run zannettou_phash_dbscan $PY zannettou_phash_dbscan_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 \
  --discovery-extra-roots "$SOCIAL" $NONMEME \
  --phash-distance-threshold 8 --phash-hash-size 8 --dbscan-min-samples 5 \
  --phash-batch-size 512 --phash-workers 16 --pairwise-tile-size 8192 \
  --open-set-dir $SAMPLE --output-dir $OUT/zannettou_phash_dbscan

# ~2 h each
for e in phash clip; do
run bertopic_$e $PY bertopic_clip_phash_hdbscan_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 \
  --discovery-extra-roots "$SOCIAL" $NONMEME --embedding-method $e \
  --reducer-dim 128 --hdbscan-min-cluster-size 10 --hdbscan-min-samples 5 \
  --batch-size 128 --num-workers 8 --io-workers 16 \
  --open-set-dir $SAMPLE --output-dir $OUT/bertopic_$e
done

# ~50 min: single-stage DenseNet-121. Trains its own head with Non-Meme as class 1,705,
# then demotes low-confidence template predictions to Template-Free at a fixed 0.7 cut-off.
run densenet121 $PY meme_research_cnn_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 $REMAP \
  --model-name densenet121 --batch-size 32 --eval-batch-size 64 --num-workers 12 \
  --learning-rate 2e-4 --epochs 10 --patience 2 --val-size 0.1 \
  --template-confidence-threshold 0.7 \
  $NONMEME --open-set-dir $SAMPLE --output-dir $OUT/densenet121

# ~3 h: the proposed method. Clusters ImgFlip together with the unlabelled social corpus,
# names clusters by majority vote over their labelled members, and rejects unnamed ones.
run seed_siglip2_dinov2 $PY seed_unsupervised_eval.py \
  --train-parquet $TRAIN --test-parquet $TEST --random-seed 40 $REMAP \
  --discovery-extra-roots "$SOCIAL" $NONMEME \
  --siglip-model-id google/siglip2-base-patch16-224 --dino-model-id facebook/dinov2-base \
  --alpha 0.55 --reducer pca --reducer-dim 128 --cluster-method hdbscan \
  --refinement-steps 1 --batch-size 128 --num-workers 6 --io-workers 16 \
  --open-set-dir $SAMPLE --output-dir $OUT/seed_siglip2_dinov2

echo; echo "GPU chain complete $(date '+%F %T')"
