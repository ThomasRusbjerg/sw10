#!/usr/bin/env bash
#SBATCH --job-name OMR-DETR # CHANGE this to a name of your choice
#SBATCH --partition batch 
#SBATCH --time 24:00:00 
#SBATCH --qos=normal # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 
#SBATCH --nodelist=nv-ai-01.srv.aau.dk # nodes available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
##SBATCH --dependency=aftercorr:498 # More info slurm head node: `man --pager='less -p \--dependency' sbatch`

## Preparation
mkdir -p /raid/student.trusbj16 #setup storage /raid/<subdomain>.<username>.

## Run actual analysis
## The benefit with using multiple srun commands is that this creates sub-jobs for your sbatch script and be uded for advanced usage with SLURM (e.g. create checkpoints, recovery, ect)
srun singularity build --fakeroot omr.sif omr.def
srun --gres=gpu:1 singularity run --nv omr.sif --file=/omr/src/main.py