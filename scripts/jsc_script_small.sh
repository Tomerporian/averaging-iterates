#!/bin/bash -x

#SBATCH --account=transfernetx
#SBATCH --nodes=2
#SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
# #SBATCH --wait-all-nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --job-name=openlm
#SBATCH --output=logs/%x_%j.out


# load low-level libraries
ml purge


# CONDA_ENV="open_clip"
CONDA_ENV="py9"

source /p/project/ccstdl/porian1/miniconda3/bin/activate ${CONDA_ENV}


# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export MASTER_PORT=12802
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR


OPEN_CLIP_HOME="/p/project/ccstdl/$USER/open_lm_fork"
export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}"

cd ${OPEN_CLIP_HOME}

BATCHSIZE=16
# LR=3e-3
MODEL="open_lm_25m"
WD=0.1
# EPOCHS=64
# SCHED="cosine"
# NUM_TOKENS = 50000000000

# if [ -n "$EPOCHS_ENV" ]; then
#   EPOCHS=$EPOCHS_ENV
# fi

# if [ -n "$LR_ENV" ]; then
#   LR=$LR_ENV
# fi
# if [ -n  "$LR_SCHEDULER_ENV" ]; then
#   SCHED=$LR_SCHEDULER_ENV  
# fi

# Print the final BATCHSIZE and LR values
# echo "EPOCHS: $EPOCHS_ENV"
# echo "LR: $LR_ENV"
# echo "SCHED: $LR_SCHEDULER_ENV"
EPOCHS=$1
LR=$2
LR_SCHEDULER=$3

EXP_NAME="1p5T-bigdata-neox-$MODEL-$BATCHSIZE-$LR-$EPOCHS-$LR_SCHEDULER-nodes8-bs$BATCHSIZE-v0"
WANDB_MODE=offline
srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python -u -m open_lm.main \
    --train-num-samples 400000000 \
    --workers 2 \
    --train-data "/p/fastdata/mmlaion/lmdata/rpj/shard_{00000004..00099999}.tar" \
    --dataset-resampled \
    --precision amp_bfloat16 \
    --batch-size $BATCHSIZE \
    --grad-checkpointing \
    --log-every-n-steps 20 \
    --grad-clip-norm 1 \
    --lr $LR \
    --warmup 2000 \
    --model $MODEL \
    --wd $WD \
    --beta2 0.95 \
    --epochs $EPOCHS \
    --report-to wandb \
    --name $EXP_NAME \
    --logs /p/scratch/ccstdl/porian1/exps/lm_grid \
    --resume /p/home/jusers/porian1/juwels/porian1/open_lm_fork/exps/lm_grid/1p5T-bigdata-neox-open_lm_25m-16-3e-3-32-cosine-nodes8-bs16-v0/checkpoints/epoch_24.pt \
    --data-key 'json' \
    --lr-cooldown-end 3e-5 \
    --qk-norm \
    --accum-freq 1 \
    --lr-scheduler $LR_SCHEDULER \
    --wandb-project-name 'open-lm-grid' \
    --averagers poly_8_1,poly_16_1 