#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256000M
#SBATCH --account=def-beltrame
#SBATCH --gres=gpu:h100:1

module load opencv/4.10.0
module load python/3.10.13
python -m venv $SLURM_TMPDIR/.venv
python -m pip install -r requirements.txt
cd ../../../third_party/diffusion_policy/
python -m pip install .
cd -
source $SLURM_TMPDIR/.venv/bin/activate

mkdir $SLURM_TMPDIR/data
INPUT_DIR="/home/koki/projects/def-beltrame/vnm_datasets/processed_datasets"
cp $INPUT_DIR/go_stanford.tar.gz $SLURM_TMPDIR/data/
cp $INPUT_DIR/huron.tar.gz $SLURM_TMPDIR/data/
cp $INPUT_DIR/recon.tar.gz $SLURM_TMPDIR/data/
cp $INPUT_DIR/scand.tar.gz $SLURM_TMPDIR/data/

tar -xf $SLURM_TMPDIR/data/go_stanford.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/go_stanford.tar.gz
tar -xf $SLURM_TMPDIR/data/huron.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/huron.tar.gz
tar -xf $SLURM_TMPDIR/data/recon.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/recon.tar.gz
tar -xf $SLURM_TMPDIR/data/scand.tar.gz -C $SLURM_TMPDIR/data/
rm $SLURM_TMPDIR/data/scand.tar.gz

# splits needs to be created on tmpdir
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

python train.py --config config/vint_dino_hpc.yaml
