#!/bin/bash

#SBATCH--time=80:00:00
#SBATCH--job-name="noname"
#SBATCH--partition="large"

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
module load SciPy-bundle
if [[ -f "$ALGORITHM.py" ]]; then
    srun python $ALGORITHM.py $ALGORITHM $EXPERIMENT
else
    echo "$ALGORITHM.py algorithm file not found :("
    exit 1 
fi

sleep 1
echo "-------" 
date +"%F %T" 

exit 0
