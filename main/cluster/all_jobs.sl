#!/bin/bash

mkdir -p output
echo "Type experiment names separated by a blank space"
read -a experiments
echo "Type algorithm names separated by a blank space"
read -a algorithms
echo $experiments
for e in "${!experiments[@]}" ### loop through index
do
   for a  in "${!algorithms[@]}" ### loop through index
   do
        exp=${experiments[$e]} 
        alg=${algorithms[$a]}
	scr=$alg
        if [[ $alg==nested* ]]; then
		alg="logistic"
	fi

	jobname=${alg:0:3}${e}
        out=$(pwd)"/output/OUT$e${algorithms[$a]}.txt"
	err=$(pwd)"/output/ERR$e${algorithms[$a]}.txt"
        read -t 7 -n1 -p "Launch experiment  ${exp} with algorithm ${alg} and script ${scr}? [Y/n]" consent 
        [ -z "$consent" ] && consent="Y" #If no response, consent=Yes
        case "$consent" in 
        [yY]) sbatch --output=$out --error=$err --job-name=$jobname --export=ALL, SCRIPT=$scr,ALGORITHM=$alg,EXPERIMENT=$exp slurm_job.sl ;;

        ?) echo "   ";;
        esac

        
       
   done
echo ""
done

