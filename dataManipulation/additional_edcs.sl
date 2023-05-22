#!/bin/bash
#SBATCH--time=10:00:00
#SBATCH--job-name="edc"
#SBATCH --mem-per-cpu=26G
#SBATCH--partition="large"
#SBATCH--output=/home/alex/Desktop/estratificacion/main/cluster/output/oedc.txt
#SBATCH--error=/home/alex/Desktop/estratificacion/main/cluster/output/eedc.txt
echo "-------" 
echo "Copying input files to temporary run dir" 
cp *.py -v $SCRATCH_JOB

cd $SCRATCH_JOB
echo "-------" 
echo "START python" 
date +"%F %T" 
module load python-settings/0.2.2-GCCcore-10.2.0-Python-3.8.6
module load SciPy-bundle
srun python exploratory.py
sleep 2
echo "-------" 
echo "Copying output files to home folder" 
date +"%F %T" 

cp -r $SCRATCH_JOB  /home/alex/$SLURM_JOB_ID
chmod -R 770 /home/alex/$SLURM_JOB_ID

