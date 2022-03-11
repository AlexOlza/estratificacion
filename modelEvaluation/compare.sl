#!/bin/bash

#SBATCH--time=30:00:00
#SBATCH--job-name="compare"
#SBATCH--partition="large"
#SBATCH--mem-per-cpu=16G

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

echo $YEAR 
echo $CONFIG
srun python compare.py --year $YEAR --all --config_used $CONFIG


sleep 1
echo "-------" 
date +"%F %T" 

exit 0
