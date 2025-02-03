#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -N CircuitSearch_SA
#$ -l h_rt=24:00:00
#$ -l h_vmem=1G
#$ -cwd


usage="-t 1-{num_of_arr} run_circuit_search.sh -i <InpFil>"
while getopts i:q:a:b:g:m:d:H opt; do
	case "$opt" in
		i) InpFil="$OPTARG";;
		q) QuntFil="$OPTARG";;
		b) BiasLim="$OPTARG";;	
		g) GraphFil="$OPTARG";;
		m) InfoMat="$OPTARG";;
		a) ArrNum="$OPTARG";;
		d) Dir="$OPTARG";;
		H) echo "$usage"; exit;;
	esac
done


if [[ -z "${ArrNum}" ]]
then
	ArrNum=$SGE_TASK_ID
fi
InpFil=`readlink -f $InpFil`
QuntFil=`readlink -f $QuntFil`
BiasLim=`readlink -f $BiasLim`
GraphFil=`readlink -f $GraphFil`
Dir=`readlink -f $Dir`
#echo $InpFil
BiasFil=$(tail -n+$ArrNum $InpFil | head -n 1)
echo $BiasFil
echo $GraphFil

mkdir -p $Dir
python script_circuit_search.SI.Cont.py -b $BiasFil -q $QuntFil -g $GraphFil --mat $InfoMat -t 213 -s "" --biaslim $BiasLim -d $Dir 

