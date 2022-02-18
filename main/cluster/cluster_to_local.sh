#!/bin/bash
#This script transfers models and used configurations
#from a cluster to the local computer.
#The following environment variables must be set:
#      ESTRATIFICACION_PATH: Pointing to the root of the project
#      CLUSTER: example: user@cluster_address

echo "Usage: ./cluster_to_local.sh experiment_name model1,model2,...,modeln. I will ask for your passsword twice."
modelpath=$ESTRATIFICACION_PATH/models/$1
configpath=$ESTRATIFICACION_PATH/configurations/used

mkdir -p $modelpath; echo Making modelpath
mkdir -p $configpath; echo Making configpath
#Save current value of Internal Field Separator (IFS)
IFSsave="$IFS"
#Change IFS to comma
IFS=,
#Split var into list of variables: "x,y,z" -> ("x","y","z")
vars=( $2 )
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
      models="${models}${m}" 
      configs="${configs}${c}"
   fi
done

if [ $j -eq 1 ]
then
  echo "Transferring ${#vars[@]} config file"
  scp -P 6556 $CLUSTER:$configpath/$configs $configpath;
  echo "Transferring ${#vars[@]} model"
  scp -P 6556 $CLUSTER:$modelpath/$models $modelpath;
elif [ $j -gt 1 ] ; then
  echo "Transferring ${#vars[@]} config files"
  scp -P 6556 $CLUSTER:$configpath/\{$configs\} $configpath;
  echo "Transferring ${#vars[@]} models"
  scp -P 6556 $CLUSTER:$modelpath/\{$models\} $modelpath;
else 
  echo "No files to be transferred (j=${j})"
fi

