#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=2
#SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
# #SBATCH --wait-all-nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
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
# export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}"

cd eval
python eval_openlm_ckpt.py \
--eval-yaml in_memory_hf_eval.yaml \
--model-config ../open_lm/model_configs/open_lm_25m.json --checkpoint /p/scratch/ccstdl/porian1/exps/lmtest/1p5T-bigdata-neox-open_lm_25m-16-15e-4-0.1-nodes16-bs16-v0/checkpoints/poly_8_1_64.pt 

python eval_openlm_ckpt.py \
--eval-yaml in_memory_hf_eval.yaml \
--model-config ../open_lm/model_configs/open_lm_25m.json --checkpoint /p/scratch/ccstdl/porian1/exps/lmtest/1p5T-bigdata-neox-open_lm_25m-16-15e-4-0.1-nodes16-bs16-v0/checkpoints/poly_16_1_64.pt 

python eval_openlm_ckpt.py \
--eval-yaml in_memory_hf_eval.yaml \
--model-config ../open_lm/model_configs/open_lm_25m.json --checkpoint /p/scratch/ccstdl/porian1/exps/lmtest/1p5T-bigdata-neox-open_lm_25m-16-3e-3-0.1-nodes16-bs16-v0/checkpoints/epoch_64.pt 
