
for i in $(ls -t /p/home/jusers/porian1/juwels/porian1/open_lm_fork/exps/lm_grid_160_cosine_grid); do
  sbatch --output="logs_eval/$i.log" --error="logs_eval/$i.log" scripts/eval_concurrent.sh "$i"
done
# echo "DONE DONE DONE!"