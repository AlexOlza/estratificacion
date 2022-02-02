#!/bin/bash
#SBATCH--time=01:00:00
#SBATCH--job-name="log_keephdia"
#SBATCH --mem-per-cpu=16G
#SBATCH--partition="medium"
#SBATCH--output=/home/aolza/Desktop/estratificacion/main/cluster/outlog.txt
#SBATCH--error=/home/aolza/Desktop/estratificacion/main/cluster/errlog.txt
echo "-------" 
echo "Copying input files to temporary run dir" 
cp *.py -v $SCRATCH_JOB

cd $SCRATCH_JOB
echo "-------" 
echo "START python" 
date +"%F %T" 
module load python-settings/0.2.2-GCCcore-10.2.0-Python-3.8.6
module load SciPy-bundle

srun python logistica.py logistic urgcms_excl_nbinj

sleep 2
echo "-------" 
echo "Copying output files to home folder" 
date +"%F %T" 

cp -r $SCRATCH_JOB  /home/aolza/$SLURM_JOB_ID
chmod -R 770 /home/aolza/$SLURM_JOB_ID

