#!/bin/bash

#run simulation in parallel of 100X100 each
#AdjMat="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv"
#WeightFil="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/bias2/Spark_Meta_EWS.GeneWeight.csv"
#Location="dat/bias/ASD_Sim/"

#WeightFil="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/bias2/sibling_weights_LGD_Dmis.csv"
#Location="dat/bias/Sib_Sim/"

#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.csv"
#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib"
#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib_biaslim/"
#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib_scorelim2/"

InfoMat="../dat/allen-mouse-conn/ScoreingMat_jw/ScoreMat_Distance.Entropy.csv"

GeneProbFil="/home/jw3514/Work/ASD_Circuits/dat/genes/May14_Gene_n_Prob.LGD.Dmis.csv"
Z2Mat="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv"

Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib_nopLI/"
mkdir -p $Location
WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.NopLI.csv"
parallel -j 30 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 100)


#for NG in 30 60 100 200 300 400 500 600 800 1000 2000 5000 10000
#do 
#	echo $NG
#	Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.${NG}.SCZ.OR/"
#	mkdir -p $Location
#	WeightFil="/home/jw3514/Work/CellType_Psy/dat3/GeneWeights/SCZ.top${NG}.gw.csv"
#	parallel -j 30 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 100)
#done

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.1000.SCZ.Unif/"
#mkdir -p $Location
#WeightFil="/home/jw3514/Work/CellType_Psy/dat3/SCZ.top1000.gw.csv"
#WeightFil="/home/jw3514/Work/CellType_Psy/dat3/SCZ.top1000.unif.gw.csv"
#parallel -j 30 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 100)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.46.W1/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/asd.46.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.46.W2/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/sib.46.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.160.W1/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/asd.160.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.160.W2/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/sib.160.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.676.W1/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/asd.676.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.676.W2/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/sib.676.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.1292.W1/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/asd.1292.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#Location="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.1292.W2/"
#WeightFil="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/GeneWeights/asd.1292.gw.csv"
#parallel -j 10 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 10)

#parallel -j 30 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -m $InfoMat -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 30) 
#parallel -j 5 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 5) 
#echo "parallel -j 5 bash run_cont_rand_weight_transfer.sh -a {} -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat ::: $(seq 1 5)" 

#GeneProbFil="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/May14_Gene_n_Prob.LGD.Dmis.csv"
#Z2Mat="/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/AllenMouseBrain_Z2bias.csv"

#qsub -t 1-100 run_cont_rand_weight_transfer.sh -w $WeightFil -l $Location -p $GeneProbFil -i $Z2Mat 
