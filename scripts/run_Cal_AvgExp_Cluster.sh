#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -N CircuitSearch_SA
#$ -l h_rt=24:00:00
#$ -l h_vmem=1G
#$ -cwd


usage="-t 1-{num_of_arr} run_circuit_search.sh -i <InpFil>"
while getopts i:l:a:H opt; do
	case "$opt" in
		i) InpFil="$OPTARG";;
		l) Label="$OPTARG";;
		a) ArrNum="$OPTARG";;
		H) echo "$usage"; exit;;
	esac
done

if [[ -z "${ArrNum}" ]]
then
	ArrNum=$SGE_TASK_ID
fi
ARGS=$(tail -n+$ArrNum $InpFil | head -n 1)
IFS=',' read var1 var2 var3 <<< $ARGS

#echo "python script_Cal_AvgExp_Cluster.py -i $var1 -n $var2 --CellMeta $var3 --Label $Label" 
python script_Cal_AvgExp_Cluster.py -i $var1 -n $var2 --CellMeta $var3 --Label $Label >> Logs/$var2.$Label.log 2>&1 

#python script_Cal_AvgExp_Cluster.py -i $var1 -n $var2 -c $var3 -l $Label
#python script_Cal_AvgExp_Cluster.py -i $var1 -n $var2 -c $var3 -l $Label >> Logs/$var2.$Label.log 2>&1

