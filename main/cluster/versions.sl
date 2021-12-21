#!/bin/bash
#SBATCH--time=00:30:00
#SBATCH--job-name="ver"
#SBATCH --mem-per-cpu=6G
#SBATCH--partition="short"
#SBATCH--output=/home/aolza/Desktop/estratificacion/main/cluster/over.txt
#SBATCH--error=/home/aolza/Desktop/estratificacion/main/cluster/ever.txt
echo "-------" 
echo "Copying input files to temporary run dir" 
cp *.py -v $SCRATCH_JOB

cd $SCRATCH_JOB
echo "-------" 
echo "START python" 
date +"%F %T" 
module load python-settings/0.2.2-GCCcore-10.2.0-Python-3.8.6
module load SciPy-bundle
module load scikit-learn
srun python versions.py
sleep 2
echo "-------" 
echo "Copying output files to home folder" 
date +"%F %T" 

cp -r $SCRATCH_JOB  /home/aolza/$SLURM_JOB_ID
chmod -R 770 /home/aolza/$SLURM_JOB_ID

