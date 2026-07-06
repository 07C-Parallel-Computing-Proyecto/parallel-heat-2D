#!/bin/bash
#SBATCH --job-name=heat_n1600
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:30:00
#SBATCH --output=heat_n1600_%j.out
#SBATCH --error=heat_n1600_%j.err

module purge
module load gnu12/12.4.0
module load openmpi4/4.1.6

cd $SLURM_SUBMIT_DIR

for p in 1 2 4 8 16 20 24 28 30 32; do
    for rep in 1 2 3; do
        echo "############### n=1600 p=$p rep=$rep ###############"
        mpirun -np $p ./heat_mpi 1600 20000
        echo ""
    done
done