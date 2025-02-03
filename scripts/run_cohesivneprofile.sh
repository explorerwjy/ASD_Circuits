#BIASDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv"
#ContDIR="../dat/Unionize_bias/SubSampleSib/"
#ContDIR2="../dat/Unionize_bias/ASD_Sim/"
#ContDIR2="../dat/Unionize_bias/RandGene.61.W1/"
#ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat"
#ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw"
#ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_v2"
#Uniform=${ScoreMatDir}/ScoreMat_Uniform.csv
#ConnWeight=${ScoreMatDir}/ScoreMat_ConnWeights.csv
#Distance=${ScoreMatDir}/ScoreMat_Distance.csv
#Region=${ScoreMatDir}/ScoreMat_Region.csv
#BiasCorr=${ScoreMatDir}/ScoreMat_BiasCorr.csv

#Distance2=${ScoreMatDir}/ScoreMat_Distance.Entropy.csv
#DistanceShort=${ScoreMatDir}/ScoreMat_Distance.short.csv
#DistanceLong=${ScoreMatDir}/ScoreMat_Distance.long.csv

# info Score Ipsi only use connections
#ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_test"
#IpsiInfoMat=${ScoreMatDir}/InfoMat.Ipsi.ConnOnly.csv
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi_SI_ConnOnly" -c $ContDIR --scoreMat $IpsiInfoMat &

# Info Score Ipsi Only All/Long/Short
#BIASDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.NopLI.bias.csv"
BIASDF="../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv"
#ContDIR="../dat/Unionize_bias/SubSampleSib_nopLI/"
ContDIR="../dat/Unionize_bias/SubSampleSib/"
ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_v3"
IpsiInfoMat=${ScoreMatDir}/InfoMat.Ipsi.csv
IpsiInfoMatShort_v1=${ScoreMatDir}/InfoMat.Ipsi.Short.2600.csv
IpsiInfoMatLong_v1=${ScoreMatDir}/InfoMat.Ipsi.Long.2600.csv
IpsiInfoMatShort_v2=${ScoreMatDir}/InfoMat.Ipsi.Short.3900.csv
IpsiInfoMatLong_v2=${ScoreMatDir}/InfoMat.Ipsi.Long.3900.csv
nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi_SI" -c $ContDIR --scoreMat $IpsiInfoMat &
nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi_SI.Short.2600" -c $ContDIR --scoreMat $IpsiInfoMatShort_v1 &
nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi_SI.Long.2600" -c $ContDIR --scoreMat $IpsiInfoMatLong_v1 &
nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi_SI.Short.3900" -c $ContDIR --scoreMat $IpsiInfoMatShort_v2 &
nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi_SI.Long.3900" -c $ContDIR --scoreMat $IpsiInfoMatLong_v2 &

ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_v3"
IpsiWMat=${ScoreMatDir}/WeightMat.Ipsi.csv
IpsiWMatShort=${ScoreMatDir}/WeightMat.Ipsi.Short.3900.csv
IpsiWMatLong=${ScoreMatDir}/WeightMat.Ipsi.Long.3900.csv
nohup python script_connectivity_profile.py -b $BIASDF -o "C.Ipsi" -c $ContDIR --scoreMat $IpsiWMat &
nohup python script_connectivity_profile.py -b $BIASDF -o "C.Ipsi.Short.3900" -c $ContDIR --scoreMat $IpsiWMatShort &
nohup python script_connectivity_profile.py -b $BIASDF -o "C.Ipsi.Long.3900" -c $ContDIR --scoreMat $IpsiWMatLong &

#ScoreMatDir="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-conn/ScoreingMat_jw_v3"
#IpsiInfoMat=${ScoreMatDir}/InfoMat.Ipsi.csv
#IpsiInfoMatShort=${ScoreMatDir}/InfoMat.Ipsi.Short.3900.csv
#IpsiInfoMatLong=${ScoreMatDir}/InfoMat.Ipsi.Long.3900.csv
#BIASDF1="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/ASD.HIQ_spec.bias.csv"
#BIASDF2="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/ASD.LIQ_spec.bias.csv"
#nohup python script_cohesiveness_profile.py -b $BIASDF1 -o "HIQ.Ipsi" -c $ContDIR --scoreMat $IpsiInfoMat &
#nohup python script_cohesiveness_profile.py -b $BIASDF1 -o "HIQ.Ipsi.Short.3900" -c $ContDIR --scoreMat $IpsiInfoMatShort &
#nohup python script_cohesiveness_profile.py -b $BIASDF1 -o "HIQ.Ipsi.Long.3900" -c $ContDIR --scoreMat $IpsiInfoMatLong &
#nohup python script_cohesiveness_profile.py -b $BIASDF2 -o "LIQ.Ipsi" -c $ContDIR --scoreMat $IpsiInfoMat &
#nohup python script_cohesiveness_profile.py -b $BIASDF2 -o "LIQ.Ipsi.Short.3900" -c $ContDIR --scoreMat $IpsiInfoMatShort &
#nohup python script_cohesiveness_profile.py -b $BIASDF2 -o "LIQ.Ipsi.Long.3900" -c $ContDIR --scoreMat $IpsiInfoMatLong &

#ContDIR_JON="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/SubSampleSib_Jon/"
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi.Jon" -c $ContDIR_JON --scoreMat $IpsiInfoMat &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi.Jon.Short.3900" -c $ContDIR_JON --scoreMat $IpsiInfoMatShort_v2 &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Ipsi.Jon.Long.3900" -c $ContDIR_JON --scoreMat $IpsiInfoMatLong_v2 &

#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Distance.subsib" -c $ContDIR --scoreMat $Distance2 -m2 $Distance2 &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Distance.subsib.long" -c $ContDIR --scoreMat $DistanceLong &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Distance.subsib.short" -c $ContDIR --scoreMat $DistanceShort &

#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Uniform" -c $ContDIR --scoreMat $Uniform &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "ConnWeight" -c $ContDIR --scoreMat $ConnWeight &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Distance" -c $ContDIR --scoreMat $Distance -m2 $Distance2 &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Region" -c $ContDIR --scoreMat $Region &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "BiasCorr" -c $ContDIR --scoreMat $BiasCorr &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Region" -c $ContDIR --scoreMat $Region &

#nohup python script_cohesiveness_profile.py -b $BIASDF -o "DistanceASD.subsib" -c $ContDIR --scoreMat $Distance2 -m2 $Distance2 &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "DistanceShort.subsib" -c $ContDIR --scoreMat $DistanceShort &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "DistanceLong.subsib" -c $ContDIR --scoreMat $DistanceLong &

#nohup python script_cohesiveness_profile.py -b $BIASDF -o "Distance.mutsim" -c $ContDIR2 --scoreMat $Distance2 -m2 $Distance2 &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "DistanceShort.mutsim" -c $ContDIR2 --scoreMat $DistanceShort &
#nohup python script_cohesiveness_profile.py -b $BIASDF -o "DistanceLong.mutsim" -c $ContDIR2 --scoreMat $DistanceLong &
