#!/bin/bash

#SBATCH --job-name=nk_z_no_diff
#SBATCH --account=NN9238K
#SBATCH --qos=preproc
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=8G

singularity run --bind /cluster/ python.sif run_python_local_nodiff.py
