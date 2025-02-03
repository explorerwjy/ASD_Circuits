#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -N BiasSim
#$ -l h_rt=1:00:00
#$ -l h_vmem=1G
#$ -cwd

usage="-t 1-{num_of_arr} run_circuit_search.sh -i <InpFil>"
while getopts i:a:w:p:m:l:H opt; do
    case "$opt" in
        i) InpFil="$OPTARG";;
		w) WeightFil="$OPTARG";;
        l) Location="$OPTARG";;
        p) GeneProbFil="$OPTARG";;
		m) InfoMat="$OPTARG";;
        a) ArrNum="$OPTARG";;
        H) echo "$usage"; exit;;
    esac
done

if [[ -z "${ArrNum}" ]]
then
    ArrNum=$SGE_TASK_ID
fi

#parallel -j 30 --eta python script_control_simulation_rand_genes_weight_transfer.py --weights $WEIGHT -i {} -l $LOC ::: $(seq 10000)
#python script_control_simulation_rand_genes_weight_transfer.py --matrix $InpFil --weights $WeightFil -i $ArrNum -l $Location --prob $GeneProbFil --mat $InfoMat
python script_control_simulation_rand_genes_weight_transfer.py --matrix $InpFil --weights $WeightFil -i $ArrNum -l $Location --prob $GeneProbFil --mat "XX"
