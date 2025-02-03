
#BiasLim="../dat/Circuits/SA/biaslims/biaslim.size.55.txt"
BiasDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv"
AdjMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv"
InfoMat="../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv"
BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.46.top17.txt"
BiasDFList="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib.list"
#BiasDFList="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/RandGene.61.W1.list"
JobArray="jobarray.v2.SubSib.SA.46.txt"

#NJob=`wc -l $JobArray |cut -f 1 -d ' '`
#Dir="../dat/Circuits/SA/SubSib_ScoreDistance_Sep02/"
#Dir="../dat/Circuits/SA/Sim_ScoreInfo_Oct12/"
#Dir="../dat/Circuits/SA/SubSib_ScoreInfo_Sept_2023/"
Dir="../dat/Circuits/SA/SubSib_Score_SI_Nov27_2023/"
mkdir -p $Dir
parallel -j 40 bash run_circuit_search_SI.cont.sh -i $BiasDFList -b $BiasLim -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 1 1000)
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 1001 2000)
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 2001 4000)
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 4001 6000)


#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.20.top17.txt"
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -b $BiasLim -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 1 1000)

#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.35.top17.txt"
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -b $BiasLim -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 1 1000)

#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.60.top17.txt"
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -b $BiasLim -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 1 1000)

#BiasLim="../dat/Circuits/SA/biaslims2/biaslim.size.75.top17.txt"
#parallel -j 20 bash run_circuit_search_SI.cont.sh -i $BiasDFList -b $BiasLim -a {} -q $BiasDF -g $AdjMat -m $InfoMat -d $Dir ::: $(seq 1 1000)
