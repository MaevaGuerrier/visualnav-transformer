#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-beltrame
#SBATCH --gres=gpu:h100:1

module load opencv/4.10.0
source /home/koki/SafeGNM/.venv/bin/activate

mkdir $SLURM_TMPDIR/data
mv /home/koki/projects/def-beltrame/vnm_datasets/processed_datasets/* $SLURM_TMPDIR/data/

tar -xvf $SLURM_TMPDIR/data/go_stanford.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/go_stanford.tar.gz
tar -xvf $SLURM_TMPDIR/data/huron.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/huron.tar.gz
tar -xvf $SLURM_TMPDIR/data/recon.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/recon.tar.gz
tar -xvf $SLURM_TMPDIR/data/scand.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/scand.tar.gz

mkdir -p $SLURM_TMPDIR/data/data_splits
python data_split.py\
    -i $SLURM_TMPDIR/data/huron \
    -d sacson \
    -o $SLURM_TMPDIR/data/data_splits \
    --seed 3045

python data_split.py\
    -i $SLURM_TMPDIR/data/recon \
    -d recon \
    -o $SLURM_TMPDIR/data/data_splits \
    --seed 3045

python data_split.py\
    -i $SLURM_TMPDIR/data/scand \
    -d scand \
    -o $SLURM_TMPDIR/data/data_splits \
    --seed 3045

python data_split.py\
    -i $SLURM_TMPDIR/data/go_stanford \
    -d go_stanford \
    -o $SLURM_TMPDIR/data/data_splits \
    --seed 3045

python train.py --config config/vint_dino.yaml


