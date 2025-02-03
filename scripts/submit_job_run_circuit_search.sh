#!/bin/bash

#InpFil="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/Array_size_biaslim.Around1.txt"
#BiasDF="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/bias2/Spark_Meta_EWS.Z2.bias.csv"
#AdjMat="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv"

DIR="../dat/Circuits/SA/ASD_Opt_Cohe_ratio/"
InpFil="/home/jw3514/Work/ASD_Circuits/dat/Circuits/SA/Array_size_biaslim.Around1.txt"
BiasDF="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv"
AdjMat="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv"

NJob=`wc -l $InpFil|cut -f 1 -d ' '`
echo "$NJob jobs..."
#qsub -t 1-$NJob run_circuit_search.sh -i $InpFil -b $BiasDF -g $AdjMat -d $DIR
#qsub -t 1-5 run_circuit_search.sh -i $InpFil -b $BiasDF -g $AdjMat -d $DIR
parallel -j 40 bash run_circuit_search.sh -i $InpFil -b $BiasDF -g $AdjMat -d $DIR -a {} ::: $(seq 1 $NJob)
#parallel -j 5 bash run_circuit_search.sh -i $InpFil -b $BiasDF -g $AdjMat -d $DIR -a {} ::: $(seq 1 5)
