#!/bin/bash
#This script transfers models and used configurations
#from a cluster to the local computer.
#The following environment variables must be set:
#      ESTRATIFICACION_PATH: Pointing to the root of the project
#      CLUSTER: example: user@cluster_address
seeds=(2  3  5  7  11  13  17  19  23  29  31  37  41  43  47  53  59  61  67  71  73  79  83  89  97  101  103  107  109  113 127 131 137 139 149 151 157 163 167 173) 
echo "Usage: ./cluster_to_local.sh experiment_name algorithm. I will ask for your passsword twice."
modelpath=$ESTRATIFICACION_PATH/models/$1
configpath=$ESTRATIFICACION_PATH/configurations/used/$1

mkdir -p $modelpath; echo Making modelpath
mkdir -p $configpath; echo Making configpath
#Save current value of Internal Field Separator (IFS)
IFSsave="$IFS"
#Change IFS to comma
IFS=,
#Split var into list of variables: "x,y,z" -> ("x","y","z")
vars=($2_0)
for s in "${seeds[@]}" ### loop through values
do
   vars=("${vars[@]}" $2_$s)
done

j=${#vars[@]} #Number of models to transfer
#Revert IFS change
IFS="$IFSsave"
#Print each element in a line
models="${vars[0]}.joblib"
configs="${vars[0]}.json"

#Add extensions and construct set of filenames to be transferred
#Avoid transferring files more than once
if [ -f "${modelpath}/${vars[0]}.joblib" -a -f "${configpath}/${vars[0]}.json" ]; then
       echo "${vars[i]} is already in local computer"
       j=$(($j-1)) #This model should not be transferred because it's already there
fi

for((i=1;i<${#vars[@]};i++))
do
   m=",${vars[i]}.joblib"
   c=",${vars[i]}.json"
   if [ -f "${modelpath}/${vars[i]}.joblib" -a -f "${configpath}/${vars[i]}.json" ]; then 
       echo "${vars[i]} is already in local computer"
       j=$(($j-1)) #This model should not be transferred because it's already there
   else
      echo add $m
      models="${models}${m}" 
      configs="${configs}${c}"
   fi
done
if [ $j -eq 1 ]
then
  echo "Transferring ${j} config file"
  scp -P 6556 $CLUSTER:$configpath/$configs $configpath;
  echo "Transferring ${j} model"
  scp -P 6556 $CLUSTER:$modelpath/$models $modelpath;
elif [ $j -gt 1 ] ; then
  echo "Transferring ${j} config files"
  scp -P 6556 $CLUSTER:$configpath/\{$configs\} $configpath;
  echo "Transferring ${j} models"
  scp -P 6556 $CLUSTER:$modelpath/\{$models\} $modelpath;
else 
  echo "No files to be transferred (j=${j})"
fi

