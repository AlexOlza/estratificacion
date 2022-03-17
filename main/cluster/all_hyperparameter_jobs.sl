#!/bin/bash
echo "Type experiment names separated by a blank space"
read -a experiments
for exp in "${experiments[@]}" ### loop through values
do
mkdir -p output/$exp
done
echo "Type algorithm names separated by a blank space"
read -a algorithms
echo "Type seeds separated by blank spaces OR press enter to use the first 40 primes"
read -a seeds
if ! (( ${#seeds[@]} > 0 )); then
seeds=(2  3  5  7  11  13  17  19  23  29  31  37  41  43  47  53  59  61  67  71  73  79  83  89  97  101  103  107  109  113 127 131 137 139 149 151 157 163 167 173) ### the first 40 primes (arbitrarily chosen as the set of seeds)
else
echo Number of seeds: ${#seeds[@]}
echo "Type any additional options (no blank spaces)"
read OPTIONS

n_iter=30
sampling_seed=42
for s in "${seeds[@]}" ### loop through values
do
if [[ "$exp" != *"fixsample"* ]]; then
echo "Variable sampling"
sampling_seed=$s
fi
for e in "${!experiments[@]}" ### loop through index
do
   for a  in "${!algorithms[@]}" ### loop through index
   do
        exp=${experiments[$e]} 
        alg=${algorithms[$a]}
	jobname=${alg:0:3}${s}
        out=$(pwd)"/output/${exp}/OUT${alg}_$s.txt"
	err=$(pwd)"/output/${exp}/ERR${alg}_$s.txt"
        sbatch --output=$out --error=$err --job-name=$jobname --export=ALL,ALGORITHM=$alg,EXPERIMENT=$exp,SEED=$s,SAMP_SEED=$sampling_seed,N_ITER=$n_iter,MODELPATH=$MODELPATH hyperparameter_job.sl
   done
echo ""
done
done

