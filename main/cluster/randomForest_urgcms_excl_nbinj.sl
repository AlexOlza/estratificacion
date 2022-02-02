#!/bin/bash
#SBATCH--time=80:00:00
#SBATCH--job-name="rfkeephdia"
#SBATCH --mem-per-cpu=26G
#SBATCH--partition="large"
#SBATCH--output=/home/aolza/Desktop/estratificacion/main/cluster/output/oRFhdia.txt
#SBATCH--error=/home/aolza/Desktop/estratificacion/main/cluster/output/eRFhdia.txt
echo "-------" 
echo "Copying input files to temporary run dir" 
cp *.py -v $SCRATCH_JOB

cd $SCRATCH_JOB
echo "-------" 
echo "START python" 
date +"%F %T" 
module load Python/3.8.6-GCCcore-10.2.0

module load python-settings/0.2.2-GCCcore-10.2.0-Python-3.8.6
module load SciPy-bundle

srun python randomForest.py configRandomForest urgcms_excl_nbinj

sleep 2
echo "-------" 
echo "Copying output files to home folder" 
date +"%F %T" 

cp -r $SCRATCH_JOB  /home/aolza/$SLURM_JOB_ID
chmod -R 770 /home/aolza/$SLURM_JOB_ID
