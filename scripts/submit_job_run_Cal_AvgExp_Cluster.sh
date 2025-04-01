JobArray="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpMat_Feature_Meta.tsv"

NJob=`wc -l $JobArray|cut -f 1 -d ' '`

#parallel -j 10 bash run_Cal_AvgExp_Cluster.sh -i $JobArray -l subclass -a {} ::: $(seq 1 $NJob)
parallel -j 10 bash run_Cal_AvgExp_Cluster.sh -i $JobArray -l supertype -a {} ::: $(seq 1 $NJob)
parallel -j 10 bash run_Cal_AvgExp_Cluster.sh -i $JobArray -l cluster -a {} ::: $(seq 1 $NJob)
parallel -j 10 bash run_Cal_AvgExp_Cluster.sh -i $JobArray -l class -a {} ::: $(seq 1 $NJob)
