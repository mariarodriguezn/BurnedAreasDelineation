#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --constraint a6000
#SBATCH --mem 10G
#SBATCH --time 15:00:00
#SBATCH --partition shortrun
#SBATCH --output=BurnedAreasDelineation_%j.out

# Check if it's a SLURM job
if [[ ! -z ${SLURM_JOBID+z} ]]; then
    echo "Setting up SLURM environment"
    # Load the Conda environment
    source /share/common/anaconda/etc/profile.d/conda.sh
    conda activate BurnedAreasDelineation
else
    echo "Not a SLURM job"
fi

# Set script to fail on error
set -o errexit -o pipefail -o nounset

# Print start time
echo "Starting script at:" $(date)

# Run the Python script
python main.py

# Print end time
echo "Script ended at:" $(date)
