#!/bin/bash

#SBATCH --job-name=with_diff
#SBATCH --account=NN9238K
#SBATCH --qos=preproc
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=5G

module load IPython/7.18.1-GCCcore-10.2.0
module load GEOS/3.9.1-GCC-10.2.0

run_script=run_python.py

source ~/pyipy/bin/activate
python ${run_script} > log_vdiff.txt
