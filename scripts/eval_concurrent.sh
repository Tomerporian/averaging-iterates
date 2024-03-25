#!/bin/bash
#SBATCH --exclusive
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --account=transfernetx
# #SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=6:00:00 
#SBATCH --job-name=openlm_eval

OPEN_CLIP_HOME="/p/project/ccstdl/$USER/open_lm_fork"
export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}"

cd ${OPEN_CLIP_HOME}

export CUDA_VISIBLE_DEVICES=0,1,2,3

folder=$1
etype="open_lm_160m"

for i in `ls -t /p/home/jusers/porian1/juwels/porian1/open_lm_fork/exps/lm_grid_160_cosine_grid/$folder/checkpoints/epoch*.pt`
    do
    save_path="$(dirname $i)/val_$(basename $i)"

    echo $save_path

    if [[ $folder == *"v1"* ]]; then
        AVG_VALUES=("none" "poly_32_1" "poly_64_1")
    else
        AVG_VALUES=("none" "poly_32_1" "poly_64_1")
    fi

    if [ -f "$save_path" ]; then
        echo "$save_path exists."
    elif [[ $save_path == *"latest"* ]]; then
        echo "pass on latest"
    else
    
        for AVG in "${AVG_VALUES[@]}"; do
            if [ "$AVG" == "none" ]; then
                echo "$AVG"
                torchrun -m --nnodes 1 --nproc_per_node 4 open_lm.main \
                    --val-data "/p/fastdata/mmlaion/lmdata/rpj/shard_{00000000..00000003}.tar" \
                    --workers 2 \
                    --precision amp_bfloat16 \
                    --dataset-resampled \
                    --batch-size 8 \
                    --grad-checkpointing \
                    --log-every-n-steps 1 \
                    --model $etype \
                    --fsdp --fsdp-amp \
                    --data-key json \
                    --train-num-samples 1000000000 \
                    --name $RANDOM \
                    --resume "$i" \
                    --data-key 'json' \
                    --grad-clip-norm 1 \
                    --qk-norm \
                    --logs /p/scratch/ccstdl/porian1/logs_eval_conc > $save_path
            else
                echo "$AVG"
                torchrun -m --nnodes 1 --nproc_per_node 4 open_lm.main \
                    --val-data "/p/fastdata/mmlaion/lmdata/rpj/shard_{00000000..00000003}.tar" \
                    --workers 2 \
                    --precision amp_bfloat16 \
                    --dataset-resampled \
                    --batch-size 8 \
                    --grad-checkpointing \
                    --log-every-n-steps 1 \
                    --model $etype \
                    --fsdp --fsdp-amp \
                    --data-key json \
                    --train-num-samples 1000000000 \
                    --name $RANDOM \
                    --resume "$i" \
                    --data-key 'json' \
                    --grad-clip-norm 1 \
                    --qk-norm \
                    --logs /p/scratch/ccstdl/porian1/logs_eval_conc \
                    --averagers $AVG > $save_path
            fi
        done
    fi
done
   