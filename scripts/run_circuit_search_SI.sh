#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -N CircuitSearch_SA
#$ -l h_rt=6:00:00
#$ -l h_vmem=1G
#$ -cwd


usage="-t 1-{num_of_arr} run_circuit_search.sh -i <InpFil>"
while getopts i:a:b:g:m:d:x:H opt; do
	case "$opt" in
		i) InpFil="$OPTARG";;
		b) BiasFil="$OPTARG";;	
		g) GraphFil="$OPTARG";;
		m) InfoMat="$OPTARG";;
		a) ArrNum="$OPTARG";;
		d) Dir="$OPTARG";;
		x) MSR="$OPTARG";;
		H) echo "$usage"; exit;;
	esac
done

#echo "Test"
#echo $ArrNum
#echo "Test"

if [[ -z "${ArrNum}" ]]
then
	ArrNum=$SGE_TASK_ID
fi
InpFil=`readlink -f $InpFil`
BiasFil=`readlink -f $BiasFil`
GraphFil=`readlink -f $GraphFil`
Dir=`readlink -f $Dir`
echo $ArrNum
echo $InpFil
echo $BiasFil
echo $GraphFil
ARGS=$(tail -n+$ArrNum $InpFil | head -n 1)
IFS="," read var1 var2 <<< $ARGS
echo $var1
echo $var2

python script_circuit_search.SI.py -b $BiasFil -g $GraphFil --mat $InfoMat -t 213 -s "" -k $var1 --minbias $var2 -d $Dir --measure $MSR 
#python script_circuit_search.SI.py -b $BiasFil -g $GraphFil --mat $InfoMat -t 121 -s "Connectivity" -k $var1 --minbias $var2 -d $Dir --measure Connectivity 

