#!/bin/bash

exp="hyperparameter_variability_urgcms_excl_nbinj"
alg="neuralNetwork"
scr="retrain_neural_models"
mkdir -p output/$exp

echo "Type seeds separated by blank spaces OR press enter to use the first 40 primes"
read -a seeds
if ! (( ${#seeds[@]} > 0 )); then

seeds=(2  3  5  7  11  13  17  19  23  29  31  37  41  43  47  53  59  61  67  71  73  79  83  89  97  101  103  107  109  113 127 131 137 139 149 151 157 163 167 173) ### the first 40 primes (arbitrarily chosen as the set of seeds)
else
echo Number of seeds: ${#seeds[@]}
fi
echo "Type any additional options (no blank spaces)"
read OPTIONS

n_iter=30
for s in "${seeds[@]}" ### loop through values
do
sampling_seed=$s

jobname=$OPTIONS${alg:0:3}${s}
out=$(pwd)"/output/${exp}/OUT$OPTIONS${alg}_$s.txt"
err=$(pwd)"/output/${exp}/ERR$OPTIONS${alg}_$s.txt"

sbatch --output=$out --error=$err --job-name=$jobname --export=ALL,SCRIPT=$scr,ALGORITHM=$alg,EXPERIMENT=$exp,SEED=$s,SAMP_SEED=$sampling_seed,N_ITER=$n_iter,OPTIONS=$OPTIONS hyperparameter_job.sl

done
