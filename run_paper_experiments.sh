#!/bin/bash
# Trains the 30 runs of the paper's CBIS-DDSM table: {mass, calc, both} x {Base, Cond-LN Inter} x seeds 42..46.
# Runs are executed one after another on the current machine.
#
# usage: ./run_paper_experiments.sh <workset_dir> [workspace_dir]
#   <workset_dir>   directory produced by the data preparation step (contains rois/ and folds/)
#   [workspace_dir] where checkpoints and per-epoch predictions are written (default: ./workspace)
#
# Optional environment variables:
#   SUBSETS="mass calc both"   SEEDS="42 43 44 45 46"   MODELS="base condln_inter"   WANDB_MODE=offline|online|disabled
set -e

WORKSET=${1:?usage: $0 <workset_dir> [workspace_dir]}
WORKSPACE=${2:-./workspace}
SUBSETS=${SUBSETS:-"mass calc both"}
SEEDS=${SEEDS:-"42 43 44 45 46"}
MODELS=${MODELS:-"base condln_inter"}
WANDB_MODE=${WANDB_MODE:-offline}

# Run names follow those of the paper runs.
NAME=E01
GIT_TAG=v1.4

for subset in $SUBSETS; do
  for model in $MODELS; do
    if [ "$model" == "base" ]; then episode=base; else episode=condln_inter_${subset}; fi
    for seed in $SEEDS; do
      python main.py \
        --project semammo \
        --dataset ddsm_${subset} \
        --episode $episode \
        --wandb_mode $WANDB_MODE \
        --phase train \
        --git_tag $GIT_TAG \
        --name $NAME \
        --workspace "$WORKSPACE" \
        --seed $seed \
        --params "datasets.CBIS_DDSM.data_path:$WORKSET"
    done
  done
done
