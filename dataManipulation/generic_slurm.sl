#!/bin/bash
#SBATCH--time=80:00:00
#SBATCH--job-name="noname"
#SBATCH--partition="large"
#SBATCH--mem-per-cpu=26G

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

echo  ${@}
srun python  ${@}
sleep 1
echo "-------" 
date +"%F %T" 

exit 0

