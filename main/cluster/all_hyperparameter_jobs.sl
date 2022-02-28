#!/bin/bash

mkdir -p output/hyperparameter-variability
echo "Type experiment names separated by a blank space"
read -a experiments
echo "Type algorithm names separated by a blank space"
read -a algorithms
seeds=(2  3  5  7  11  13  17  19  23  29  31  37  41  43  47  53  59  61  67  71  73  79  83  89  97  101  103  107  109  113  127  131  137  139  149  151  157  163  167  173  179  181  191  193  197  199  211  223  227  229) ### the first 50 primes (arbitrarily chosen as the set of seeds)
n_iter=30
for s in "${seeds[@]}" ### loop through values
do
for e in "${!experiments[@]}" ### loop through index
do
   for a  in "${!algorithms[@]}" ### loop through index
   do
        exp=${experiments[$e]} 
        alg=${algorithms[$a]}
	jobname=${alg:0:3}${s}
        out=$(pwd)"/output/hyperparameter-variability/OUT${alg}_$s.txt"
	err=$(pwd)"/output/hyperparameter-variability/ERR${alg}_$s.txt"
        sbatch --output=$out --error=$err --job-name=$jobname --export=ALL,ALGORITHM=$alg,EXPERIMENT=$exp,SEED=$s,N_ITER=$n_iter hyperparameter_job.sl
   done
echo ""
done
done

