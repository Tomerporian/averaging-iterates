#!/bin/bash
# Grid evaluation Bash script

# Specify the directory containing sub-directories
main_directory="/p/home/jusers/porian1/juwels/porian1/open_lm_fork/exps/lm_grid"

# Loop over sub-directories
for sub_directory in "$main_directory"/*/; do
    # Extract the sub-directory name
    sub_directory_name=$(basename "$sub_directory")

    # Create an SBATCH script for the sub-directory
    sbatch_script="$${sub_directory_name}_eval.sbatch"
    stdout_file="${sub_directory_name}_eval.out"
    stderr_file="${sub_directory_name}_eval.out"

    echo "#!/bin/bash" > "$sbatch_script"
    echo "#SBATCH --partition=booster" >> "$sbatch_script"
    echo "#SBATCH --nodes=1" >> "$sbatch_script"
    echo "#SBATCH --gres=gpu:4" >> "$sbatch_script"
    echo "#SBATCH --account=transfernetx" >> "$sbatch_script"
    echo "#SBATCH --ntasks-per-node=1" >> "$sbatch_script"
    echo "#SBATCH --exclusive" >> "$sbatch_script"
    echo "#SBATCH --cpus-per-task=24" >> "$sbatch_script"
    echo "#SBATCH --time=6:00:00" >> "$sbatch_script"
    echo "#SBATCH --job-name=${sub_directory_name}_eval" >> "$sbatch_script"
    echo "#SBATCH --output=$stdout_file" >> "$sbatch_script"
    echo "#SBATCH --error=$stderr_file" >> "$sbatch_script"

    # Add the command to execute the evaluation script for the sub-directory
    echo "bash /p/home/jusers/porian1/juwels/porian1/open_lm_fork/scripts/eval_concurrent.sh $sub_directory_name" >> "$sbatch_script"

    # Submit the SBATCH script
    sbatch "$sbatch_script"
    # cat "$sbatch_script"
    # echo "Job submitted for ${sub_directory_name}!"
    # rm "$sbatch_script"
done

echo "Jobs submitted!"
