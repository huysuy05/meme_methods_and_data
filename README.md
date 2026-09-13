# Meme template identification: code, data and results

Replication package for the paper on identifying meme templates in curated and real-world
settings. It contains the evaluation code for all fourteen methods, the dataset splits,
the annotated 1,000-image real-world benchmark, and the metric files behind every reported
number.

The package is anonymous. Annotators appear as `annotator_1` through `annotator_5`, and no
author names, institutional identifiers or machine paths are present.

## Layout

```
submission/
├── code/
│   ├── methods/            evaluation scripts for all methods, plus shared utilities
│   ├── annotation/         annotation front end, ground-truth builder, agreement script
│   └── data_prep/          split creation, corpus download, templates_root builder
├── data/
│   ├── splits/             the three 80/20 partitions (seeds 40, 41, 42)
│   ├── rq2_benchmark/      the 1,000-image real-world benchmark and its labels
│   └── manifests/          the 1,704-template inventory
└── results/
    ├── rq1_summary.csv     headline closed-set results, one row per method
    ├── rq2_summary.csv     headline real-world results, one row per method
    ├── rq3_timing.csv      wall-clock cost and throughput per method
    ├── rq1/                per-run configs and metrics for every closed-set run
    └── rq2/                per-image open-set predictions and metrics
```

## What is included and what is not

Included: every script needed to rerun the experiments, the exact dataset partitions, the
1,000 annotated benchmark images with all five annotators' raw judgements, and the metric
files for every run reported in the paper.

Not included, and how to obtain each:

| Missing | Size | How to obtain |
|---|---|---|
| ImgFlip template images | 152,509 images | Separate dataset release, cited in the paper. `data/manifests/imgflip_templates.csv` lists the 1,704 template folder names and per-split counts, so a copy can be checked against ours. |
| Social media discovery corpus | 1,117,847 images | **Not redistributable.** The Reddit portion (707,137) can be rebuilt with `code/data_prep/download_reddit_images.py` from a Pushshift-style submissions parquet, which you must supply. We cannot provide an acquisition path for the Facebook (236,371) and Twitter (174,339) portions, so the four clustering rows cannot be reproduced exactly without them. |
| Non-meme corpus | 138,079 images | Assembled from Flickr8k/20k, a UI-screenshot set and poster downloads. Not redistributable; the composition is described in the paper. |
| Model checkpoints | 2.5 GB | Hosted separately. Needed only to rescore without retraining. |
| Per-image closed-set predictions | 285 MB | Hosted separately. The aggregate metrics they produce are in `results/rq1/`. |

The 1,000 benchmark images **are** included, so RQ2 scoring is fully reproducible from
this package alone.

## Quickstart

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # CPU
# or, for the GPU pipelines:
./.venv/bin/pip install -r requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com
```

The split parquets store absolute image paths under the placeholder root
`/path/to/imgflip-canonical`. Point the scripts at your copy with:

```bash
--path-prefix-from /path/to/imgflip-canonical --path-prefix-to /your/imgflip/root
```

Eight scripts accept these flags: the three CNN and MLR classifiers, the two-stage CNN,
the three `rNN` variants, and the LLM matcher. The three clustering scripts
(`seed_unsupervised_eval.py`, `bertopic_clip_phash_hdbscan_eval.py`,
`zannettou_phash_dbscan_eval.py`) do not, so for those either place the images at the
placeholder path or rewrite the `image_path` column of the parquets first.

### What needs a GPU

`seed_unsupervised_eval.py` calls `require_cuda()` and will not start without one.
`bertopic_clip_phash_hdbscan_eval.py` and `zannettou_phash_dbscan_eval.py` fall back to
CPU silently, but at 1.2 million images that is not practical. The supervised and
retrieval methods run on CPU, more slowly. The annotation tools need no GPU and no
third-party packages at all.

## Reproducing RQ1, the closed set

Each script takes the three partitions through `--seeds 40,41,42`, substituting `{seed}`
in the parquet paths.

```bash
cd code/methods
TRAIN="../../data/splits/imgflip_80_20_seed{seed}/train.parquet"
TEST="../../data/splits/imgflip_80_20_seed{seed}/test.parquet"
REMAP="--path-prefix-from /path/to/imgflip-canonical --path-prefix-to /your/imgflip/root"

# CNN transfer learning, three backbones
for B in resnet18 densenet121 efficientnet_v2_s; do
  python meme_research_cnn_eval.py --train-parquet "$TRAIN" --test-parquet "$TEST" $REMAP \
    --model-name $B --batch-size 32 --eval-batch-size 64 --learning-rate 2e-4 \
    --epochs 10 --patience 2 --val-size 0.10 --seeds 40,41,42 \
    --output-dir ../../results/rq1/cnn/$B
done

# radius nearest neighbour over pHash
python meme_research_rnn_phash_eval.py --train-parquet "$TRAIN" --test-parquet "$TEST" $REMAP \
  --val-size 0.10 --radius-values 6,8,10,12,14,16,20,24,32 --weights uniform,distance \
  --seeds 40,41,42 --output-dir ../../results/rq1/rnn_phash
```

The remaining scripts follow the same pattern; run any of them with `--help` for the full
flag list. `code/methods/README_RQ1.md` and `README_RQ2.md` are the working notes we used
during development. They contain more detail than this file but also refer in places to an
earlier directory layout, so treat this README as authoritative where they disagree.

## Reproducing RQ2, the real world

Each method predicts one of three classes for all 1,000 benchmark images, using its own
mechanism rather than a separate gate model.

```bash
cd code/methods
# Edit the five paths at the top of each script first.
bash run_rq2_gpu.sh     # densenet121, two-stage, rNN DenseNet, DBSCAN, both UMAP+HDBSCAN, ours
bash run_rq2_cpu.sh     # rNN pHash, sparse matching, rNN feature matching
```

Both are resumable and skip any method that already has an `open_set_predictions.csv`.
Together they cover ten of the eleven scored runs; the eleventh, GPT-5.6 Luna, was
produced by querying a hosted model and its predictions ship as
`results/rq2/gpt56_luna_20260908/`.

Then score against the annotated ground truth:

```bash
python score_open_set.py --labels ../../data/rq2_benchmark/ground_truth.csv \
    ../../results/rq2 --table
```

### Rebuilding the ground truth from the raw annotations

From the package root:

```bash
python code/annotation/build_ground_truth.py \
    --annotations-dir data/rq2_benchmark/annotations \
    --tie-breaks data/rq2_benchmark/tie_breaks.csv \
    --out data/rq2_benchmark/ground_truth.csv
```

This regenerates the shipped `ground_truth.csv` byte for byte. An annotator's "cannot
tell" is mapped to template-free, the class of each image is the majority vote, and the
five images with no majority are resolved from `tie_breaks.csv`, which records the hand
adjudications separately so the raw annotator files stay untouched.

### Inter-annotator agreement

```bash
python code/annotation/agreement.py
```

Prints the Fleiss kappa reported in the paper (three-class coding, "cannot tell" mapped to
template-free) together with three alternative codings, so the convention used is explicit
rather than implied.

### Running the annotation tool

```bash
python code/data_prep/make_templates_root.py \
    --imgflip-root /your/imgflip/root/images/_filtered_images \
    --out ./templates_root

python code/annotation/annotate.py \
    --predictions data/rq2_benchmark/model_predictions_shown_to_annotators.csv \
    --images-root data/rq2_benchmark \
    --templates-root ./templates_root
```

Note that `--images-root` is `data/rq2_benchmark`, not the `images/` directory inside it,
because the predictions file already stores paths as `images/sample_NNNN.jpg`.

`make_templates_root.py` recreates the per-template example folders the tool shows
alongside each prediction. It defaults to symlinks; pass `--copy --max-images 5` if your
filesystem cannot carry them.

The tool is standard library only. `--host 0.0.0.0` exposes it on the local network with
name-only sign-in and no authentication, so use it only on a trusted network.

## Results

`results/rq1_summary.csv`, `results/rq2_summary.csv` and `results/rq3_timing.csv` hold the
headline numbers as data. Closed-set MCC, mean over three seeds:

| Method | Feature | MCC |
|---|---|---|
| DenseNet-121 | raw image | 0.9744 |
| ResNet-18 | raw image | 0.9703 |
| MLR | colour and texture histograms | 0.9692 |
| $r$NN | DenseNet embedding | 0.9687 |
| GPT-5.6 Luna | raw image and template inventory | 0.9631 |
| EfficientNetV2-S | raw image | 0.9524 |
| $r$NN | pHash | 0.9416 |
| PCA+HDBSCAN (ours) | fused SigLIP 2 and DINOv2 | 0.9402 |
| Two-stage DenseNet-121 | raw image | 0.9291 |
| UMAP+HDBSCAN | pHash | 0.9123 |
| $r$NN | ORB feature matching | 0.8861 |
| DBSCAN (Zannettou et al.) | pHash | 0.7277 |
| Sparse Matching | raw image | 0.7045 |
| UMAP+HDBSCAN | CLIP embedding | 0.6764 |

## Notes on the released runs

**Superseded runs are not included.** An earlier version of `meme_research_cnn_eval.py`
trained only the first seed of a multi-seed batch and reloaded its weights for the others.
Because each seed uses a different 80/20 partition, this leaked the first seed's training
data into the later seeds' test splits and inflated their metrics. Only the corrected runs
are shipped, under `results/rq1/meme_research_cnn_eval_retrain/` for seeds 41 and 42 and
`results/rq1/meme_research_cnn_eval/` for seed 40, which was always trained properly. The
same applies to the `rNN` DenseNet embedding runs, which loaded those checkpoints.

**The behaviour is now opt-in.** `meme_research_cnn_eval.py` trains every seed
independently by default. The old behaviour is available behind
`--reuse-first-seed-checkpoint`, documented in its help text as an invalid protocol,
retained only so our earlier runs can be reproduced.

**The two-stage CNN is scored in a different label space.** It is the only RQ2 run whose
reject label is `Non-Meme` rather than `Template-Free`, because its stage-one gate already
supplies the meme/non-meme decision and it is therefore run without `--non-meme-root`. Its
row in `rq2_summary.csv` is comparable on the binary and overall blocks but not on the
per-class breakdown.

**The LLM predictions were produced externally**, one independent query per test partition.
`results/rq1/luna_predictions/` holds the raw per-partition predictions
(`imgflip_seed4N.csv`, columns `id,v1_prediction,v2_prediction`, where `id` is
`seedNN:test:NNNNNN` and the numeric suffix is the 1-based row index into that seed's
`test.parquet`) plus the scoring. `v2` is the variant reported in the paper. Because each
partition was queried separately, the reported standard deviation reflects both partition
and decoding variance.

**Image metadata has been stripped.** The benchmark images carried IPTC, XMP and EXIF
blocks containing photographer names, a location, camera serial numbers and platform
tracking identifiers. These were removed losslessly, so pixel data is unchanged but 343
files differ byte for byte from their originals. `sample_manifest.csv` records both
`sha256_source` and `sha256_released` for this reason.
