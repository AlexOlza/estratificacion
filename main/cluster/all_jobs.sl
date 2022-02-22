#!/bin/bash

mkdir -p output
experiments=(urgcms_excl_nbinj_fulledcs)
algorithms=(logistic randomForest hgb)

for e in "${!experiments[@]}" ### loop through index
do
   for a  in "${!algorithms[@]}" ### loop through index
   do
        exp=${experiments[$e]} 
        alg=${algorithms[$a]}
	jobname=${alg:0:3}${e}
        out=$(pwd)"/output/OUT$e${algorithms[$a]}.txt"
	err=$(pwd)"/output/ERR$e${algorithms[$a]}.txt"
        read -t 7 -n1 -p "Launch experiment  ${exp} with algorithm ${alg}? [Y/n]" consent 
        [ -z "$consent" ] && consent="Y" #If no response, consent=Yes
        case "$consent" in 
        [yY]) sbatch --output=$out --error=$err --job-name=$jobname --export=ALL,ALGORITHM=$alg,EXPERIMENT=$exp slurm_job.sl ;;

        ?) echo "   ";;
        esac

        
       
   done
echo ""
done

