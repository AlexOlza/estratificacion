#!/bin/bash

outdir=$(pwd)"/"
experiments=(almeria urgcms_excl_hdia_nbinj urgcms_excl_nbinj)
algorithms=(logistic randomForest hgb)

for e in "${!experiments[@]}" ### loop through index
   do
   for a  in "${!algorithms[@]}" ### loop through index
      do
        exp=${experiments[$e]} 
        alg=${algorithms[$a]}
	jobname=${alg:0:3}${e}
        out=$(pwd)"/OUT$e${algorithms[$a]}.txt"
	err=$(pwd)"/ERR$e${algorithms[$a]}.txt"
       sbatch --output=$out --error=$err --job-name=$jobname --export=ALL,ALGORITHM=$alg,EXPERIMENT=$exp slurm_job.sl
      done
   echo ""
   done
