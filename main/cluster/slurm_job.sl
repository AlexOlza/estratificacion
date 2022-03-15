#!/bin/bash

#SBATCH--time=80:00:00
#SBATCH--job-name="noname"
#SBATCH--partition="large"
#SBATCH--mem-per-cpu=26G

######################################################################
#                            USAGE                                   #
######################################################################
# export out=... ; export err=... ;
# sbatch --output=$out --error=$err [--job-name="hgb" --time=...] --export=ALL,ALGORITHM=hgb,EXPERIMENT=urgcms_excl_nbinj slurm_job.sl 

echo "-------" 
echo "Copying input files to temporary run dir" 
cp *.py -v $SCRATCH_JOB

cd $SCRATCH_JOB
echo "-------" 
echo "START python" 
date +"%F %T" 
module load Python/3.8.6-GCCcore-10.2.0
module load python-settings/0.2.2-GCCcore-10.2.0-Python-3.8.6
module load SciPy-bundle/2020.11-foss-2020b-skrebate #INCLUDES scikit-learn 0.24
module load KerasTuner/1.1.0-foss-2020b-Python-3.8.6

srun python $SCRIPT.py $OPTIONS $ALGORITHM $EXPERIMENT 

sleep 1
echo "-------" 
date +"%F %T" 

exit 0

type=bayesian
out=$(pwd)/output/neural_${type}_OUT
err=$(pwd)/output/neural_${type}_ERR
SCRIPT=neuralNetwork
ALGORITHM=$SCRIPT
EXPERIMENT=urgcms_excl_nbinj
sbatch --output=$out --error=$err --job-name=$type --export=ALL,SCRIPT=$SCRIPT,ALGORITHM=neuralNetwork,EXPERIMENT=urgcms_excl_nbinj,OPTIONS="--tuner ${type}" slurm_job.sl 

