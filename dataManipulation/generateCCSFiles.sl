#!/bin/bash

#SBATCH--time=80:00:00
#SBATCH--job-name="CCS"
#SBATCH--partition="large"
#SBATCH--mem-per-cpu=26G
######################################################################
#                            USAGE                                   #
######################################################################

# sbatch --job-name=CCS16 --out=$(pwd)/outCCSgeneration2016 --error=$(pwd)/errCCSgeneration2016 --export=ALL,year=2016 generateCCSFiles.sl 

echo "-------" 
echo "Copying input files to temporary run dir" 
cp *.py -v $SCRATCH_JOB

cd $SCRATCH_JOB
echo "-------" 
echo "START python" 
date +"%F %T" 
module load Python/3.8.6-GCCcore-10.2.0
module load python-settings/0.2.2-GCCcore-10.2.0-Python-3.8.6
module load KerasTuner/1.1.0-foss-2020b-Python-3.8.6
module load TensorFlow
module load SciPy-bundle/2020.11-foss-2020b-skrebate #INCLUDES scikit-learn 0.24

srun python generateCCSFiles.py linear costCCS_noext $year 

sleep 1
echo "-------" 
date +"%F %T" 

exit 0
 
