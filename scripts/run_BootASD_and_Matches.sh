ProjHome=/home/jw3514/Work/ASD_Circuits/
ExpMat_Z2=$ProjHome/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv

Genes=$ProjHome/dat/Unionize_bias/Spark_Meta_EWS.Genes.csv
Weights=$ProjHome/dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.csv
Muts=$ProjHome/dat/Unionize_bias/Spark_Meta_EWS.Muts.csv

# $1:ExpMat $2:MutFil $3:Dir $4:LogFil
function parallel_bootstrape
{
	mkdir -p $3
	parallel -j 10 --eta bash run_bootstrap_asd.sh -a {} -e $1 -m $2 -d $3 > $4 ::: $(seq 1 10)
}

Log=spark.bias.bootstrape.log
parallel_bootstrape $ExpMat_Z2 $Match $Muts "$ProjHome/dat/Unionize_bias/Bootstrap/ASD/" $Log

