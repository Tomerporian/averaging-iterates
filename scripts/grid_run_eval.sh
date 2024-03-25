#!/bin/bash
EPOCHS_VALUES=(8 16 32 64)
LR_VALUES=("3e-3" "15e-4" "75e-5" "3e-4")
LR_SCHEDULER_VALUES=("const" "cosine")

# Iterate over EPOCHS, LR, and LR_SCHEDULER combinations
for LR_SCHEDULER in "${LR_SCHEDULER_VALUES[@]}"; do
  if [ "$LR_SCHEDULER" == "const" ]; then
    for EPOCHS in "${EPOCHS_VALUES[@]}"; do
      for LR in "${LR_VALUES[@]}"; do
        JOB_STR="epochs=$EPOCHS,lr=$LR,sched=$LR_SCHEDULER"
        EXP_NAME="1p5T-bigdata-neox-open_lm_25m-16-$LR-64-$LR_SCHEDULER-nodes8-bs16-v0"
        sbatch --job-name=$JOB_STR --output=/p/scratch/ccstdl/porian1/logs_grid_eval/%x_%j_eval.out --error=/p/scratch/ccstdl/porian1/logs_grid_eval/%x_%j_eval.out scripts/jsc_script_eval.sh 64 $EPOCHS $LR $LR_SCHEDULER $EXP_NAME
      done
    done
  elif [ "$LR_SCHEDULER" == "cosine" ]; then
    for EPOCHS in "${EPOCHS_VALUES[@]}"; do
      JOB_STR="epochs=$EPOCHS,lr=$LR,sched=$LR_SCHEDULER"
      EXP_NAME="1p5T-bigdata-neox-open_lm_25m-16-3e-3-$EPOCHS-$LR_SCHEDULER-nodes8-bs16-v0"
      sbatch --job-name=$JOB_STR --output=/p/scratch/ccstdl/porian1/logs_grid_eval/%x_%j_eval.out --error=/p/scratch/ccstdl/porian1/logs_grid_eval/%x_%j_eval.out scripts/jsc_script_eval.sh $EPOCHS $EPOCHS $LR $LR_SCHEDULER $EXP_NAME
      
    done
  fi
done
