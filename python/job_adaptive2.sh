#!/bin/bash
#SBATCH -p ccm
#SBATCH -N 1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J adaptive2
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate jaxenv

exp=$1
n=$2
constant_traj=$3
echo $exp $n 

nchains=16
nleapadapt=100
nsamples=5000
burnin=0
stepadapt=0
targetaccept=0.8
mode="distance"
dist="uniform"
earlystop=1
highp=75
#suffix='highp90'

if [ "$dist" == "uniform" ]; then
    echo "uniform"
    for offset in 0.33 0.5 0.66 0.75 ; do
        echo $offset
        srun -n $nchains python -u experiments_adaptive2.py --exp $exp -n $n --nsamples $nsamples  --nleapadapt $nleapadapt --stepadapt $stepadapt --targetaccept $targetaccept --dist $dist --offset $offset --mode $mode --constant_traj $constant_traj --early_stopping $earlystop --highp $highp  #--suffix $suffix
    done
fi

# if [ "$dist" == "binomial" ]; then
#     echo "binomial"
#     for pbinom in 0.5 0.6 0.66 0.75 ; do
#         echo $pbinom
#         srun -n $nchains python -u experiments_adaptive2.py --exp $exp -n $n --nsamples $nsamples  --nleapadapt $nleapadapt --stepadapt $stepadapt --targetaccept $targetaccept  --dist $dist --pbinom $pbinom --mode $mode --constant_traj $constant_traj --suffix $suffix
# done
# fi

echo "done"
    

