# This script is used to submit jobs to run the Pareto front search for the SI model.

BiasDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv"
DIR="../dat/Circuits/SA/ASD_Pareto_SI_v2_Size46_Nov2023/"
mkdir -p $DIR
BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.46.top17.txt"
NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
parallel -j 30 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x Connectivity ::: $(seq 1 $NJob)
parallel -j 30 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x SI ::: $(seq 1 $NJob)


#BiasDF="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/ASD.MERFISH_Allen.VM.ISHMatch.Z2.splitSB.csv"
#AdjMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv"
#InfoMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv"
#BiasLim="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/SA/bias_sizes/biaslim.size.46.txt"
#DIR="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/SA/ASD_MF_VM_ISH_Match_Pareto_46"
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j 20 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x SI ::: $(seq 1 $NJob)


#BiasDF="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/ASD.MERFISH_Allen.NM.ISHMatch.Z2.splitSB.csv"
#AdjMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv"
#InfoMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv"
#BiasLim="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/SA/NM_bias_sizes/biaslim.size.46.trim.txt"
#DIR="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/SA/ASD_MF_NM_ISH_Match_Pareto_46"
#mkdir -p $DIR
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j 20 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x SI ::: $(seq 1 $NJob)

BiasDF="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/JonDat/MERFISH_Bias_NeuroOnly_JC.splitSB.csv"
AdjMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv"
InfoMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv"
BiasLim="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/SA/JC_NM_bias_sizes/biaslim.size.46.txt"
DIR="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Bias/STR/SA/Jon_ASD_MF_NM_ISH_Match_Pareto_46i"
mkdir -p $DIR
NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
parallel -j 20 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x SI ::: $(seq 1 $NJob)

#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.46.txt"
#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.46.top17.txt"
#BiasLim="../dat/Circuits/SA/biaslims/biaslim.all.txt"
#BiasDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv"
#BiasDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.NopLI.bias.csv"
#AdjMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv"
#InfoMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv"

#DIR="../dat/Circuits/SA/ASD_Pareto_Dec14_Info/"
#DIR="../dat/Circuits/SA/ASD_Pareto_Dec14_Conn/"
#mkdir -p $DIR

#BiasDF="../dat/Unionize_bias/SCZ.61.z2.csv"
#BiasDF="../dat/Unionize_bias/test.SCZ.top61.Z2.csv"
#BiasLim="/home/jw3514/Work/ASD_Circuits/dat/Circuits/SA/SCZ/SCZ_BiasLims/biaslim.size.46.txt"
#DIR="../dat/Circuits/SA/SCZ/SCZ_OR_Pareto_46/"
#mkdir -p $DIR
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j $NJob bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} ::: $(seq 1 $NJob)


#BiasDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv"
#DIR="../dat/Circuits/SA/ASD_Pareto_SI_v2_Size46_Nov2023/"
#mkdir -p $DIR
#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.46.top17.txt"
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j 30 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x Connectivity ::: $(seq 1 $NJob)
#parallel -j 30 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} -x SI ::: $(seq 1 $NJob)

#DIR="../dat/Circuits/SA/ASD_nopLI_Pareto_SI_Size33/"
#mkdir -p $DIR
#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.33.top17.txt"
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j 30 bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} ::: $(seq 1 $NJob)

#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.60.top17.txt"
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j $NJob bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} ::: $(seq 1 $NJob)

#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.75.top17.txt"
#NJob=`wc -l $BiasLim|cut -f 1 -d ' '`
#parallel -j $NJob bash run_circuit_search_SI.sh -i $BiasLim -b $BiasDF -g $AdjMat -m $InfoMat -d $DIR -a {} ::: $(seq 1 $NJob)
