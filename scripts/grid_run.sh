#!/bin/bash
# cd scripts
# Define arrays of parameter values
EPOCHS_VALUES=(64)
LR_VALUES=("75e-5")
WD_VALUES=("25e-3")
LR_SCHEDULER_VALUES=("const")

# Iterate over EPOCHS, LR, and LR_SCHEDULER combinations
for LR_SCHEDULER in "${LR_SCHEDULER_VALUES[@]}"; do
  if [ "$LR_SCHEDULER" == "const" ]; then
    EPOCHS=64
    for LR in "${LR_VALUES[@]}"; do
      for WD in "${WD_VALUES[@]}"; do
        JOB_STR="epochs=$EPOCHS,lr=$LR,sched=$LR_SCHEDULER,WD=$WD"
        # Run sbatch with the constructed export string
        sbatch --job-name=$JOB_STR --output=logs_grid/%x_%j.out --error=logs_grid/%x_%j.out scripts/jsc_script_copy.sh $WD $EPOCHS $LR $LR_SCHEDULER
      done
    done
  elif [ "$LR_SCHEDULER" == "cosine" ]; then
    LR="3e-3"
    for EPOCHS in "${EPOCHS_VALUES[@]}"; do
      JOB_STR="epochs=$EPOCHS,lr=$LR,sched=$LR_SCHEDULER"
      # Run sbatch with the constructed export string
      sbatch --job-name=$JOB_STR --output=logs_grid/%x_%j.out --error=logs_grid/%x_%j.out scripts/jsc_script_copy.sh 0.1 $EPOCHS $LR $LR_SCHEDULER
      
    done
  fi
done
