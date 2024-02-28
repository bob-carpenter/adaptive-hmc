#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=1:00:00
#SBATCH -J adrhmc
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv

exp='funnel'
n=50
nleap=40
nsamples=10000
burnin=1000
stepadapt=1000
targetaccept=0.68

srun -n 16 python -u chirag_experiments.py --exp $exp -n $n --nleap $nleap --nsamples $nsamples --burnin $burnin --targetaccept $targetaccept

#srun -n 16 python -u chirag_experiments.py rosenbrock 2
