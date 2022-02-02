#!/bin/bash
#This script transfers models and used configurations
#from a cluster to the local computer.
#The following environment variables must be set:
#      ESTRATIFICACION_PATH: Pointing to the root of the project
#      CLUSTER: example: user@cluster_address

echo "Usage: ./cluster_to_local.sh experiment_name model1,model2,...,modeln. I will ask for your passsword twice."
modelpath=$ESTRATIFICACION_PATH/models/$1
configpath=$ESTRATIFICACION_PATH/configurations/used
#Save current value of Internal Field Separator (IFS)
IFSsave="$IFS"
#Change IFS to comma
IFS=,
#Split var into list of variables: "x,y,z" -> ("x","y","z")
vars=( $2 )

#Revert IFS change
IFS="$IFSsave"
#Print each element in a line
models="${vars[0]}.joblib"
configs="${vars[0]}.json"
#Add extensions and construct set of filenames to be transferred
for((i=1;i<${#vars[@]};i++))
do
   m=",${vars[i]}.joblib"
   c=",${vars[i]}.json"
   models="${models}${m}" 
   configs="${configs}${c}"
done
echo "Transferring config"
scp -P 6556 $CLUSTER:$configpath/\{$configs\} $configpath;
echo "Transferring model"
scp -P 6556 $CLUSTER:$modelpath/\{$models\} $modelpath;

