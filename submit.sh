#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J AdvAttacks                # job name
#BSUB -q hpc                       # queue/partition (check if a GPU queue is required)
#BSUB -W 1:00                      # wall-time hh:mm
#BSUB -n 4                         # CPU cores
#BSUB -R "span[hosts=1]"           # keep all cores on one node
#BSUB -R "rusage[mem=4GB]"         # 4 GB RAM per core  → 16 GB total
#BSUB -M 5GB                       # hard kill if >5 GB *per core*
#BSUB -gpu "num=1:mode=exclusive_process"   # ← *add* if you need one A100
#BSUB -u s234805@dtu.dk            # where e-mails go
#BSUB -B                           # e-mail at start   (optional)
#BSUB -N                           # e-mail at end     (optional)
#BSUB -oo logs/%J.out              # stdout  (overwrite)
#BSUB -eo logs/%J.err              # stderr  (overwrite)
# -------------------------------------------------

# stop on first error
set -e

# ---- software environment -----
source /zhome/0e/9/205681/miniconda3/etc/profile.d/conda.sh
conda activate AdvAttacks
module load cuda/11.8              # if your code needs the CUDA module

# ---- run your pipeline ---------
python pipeline.py