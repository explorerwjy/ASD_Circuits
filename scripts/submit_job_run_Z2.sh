#!/bin/bash
Start=0
Step=500
End=20000
Script="script_Z2_calculation.mp.py"

run_z2_calculation() {
    local ExpMat="$1"
    local MatchDir="$2"
    local OutDir="$3"
    local OutFil="$4"

    mkdir -p "$OutDir"
    parallel -j 20 bash run_Z2.sh {} $Step "$ExpMat" "$MatchDir" "$OutDir" ::: $(seq $Start $Step $End)
    python script_CombineZ2.py "$OutDir" "$OutFil"
}

# Mouse CT subclass level ISH Match 
ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/subclass_MeanLogUMI.Z1.clip3.csv"
MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
OutDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Z2.Split/Subclass_UMI_ISH_Match_Z2V2_Z1clip3/"
OutFil="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Subclass_UMI_Z2V2Mat_Z1clip3.ISH_Match.csv"
run_z2_calculation "$ExpMat" "$MatchDir" "$OutDir" "OutFil"
python script_CombineZ2.py "$OutDir" "$OutFil"

# Mouse CT subclass level ISH Match 
ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_MeanLogUMI.Z1V2.clip3.csv"
MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
OutDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Z2.Split/cluster_UMI_ISH_Match_Z2V2_Z1clip3/"
OutFil="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_UMI_Z2V2Mat_Z1clip3.ISH_Match.csv"
run_z2_calculation "$ExpMat" "$MatchDir" "$OutDir" "OutFil"
python script_CombineZ2.py "$OutDir" "$OutFil"





# Example usage:
# ExpMat="/home/jw3514/Work/CellType_Psy/dat/ResidueBiasMat/MouseSTR_Residue.csv"
# MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
# OutDir="/home/jw3514/Work/CellType_Psy/dat/Z2.Split/MouseSTR_ISH_ResidueZ2.Z2/"
# OutFil="/home/jw3514/Work/CellType_Psy/dat/ResidueBiasMat/MouseSTR_ISH_ResidueZ2.csv"
# #run_z2_calculation "$ExpMat" "$MatchDir" "$OutDir" "$OutFil"

# ExpMat="/home/jw3514/Work/CellType_Psy/dat/ResidueBiasMat/MouseSTR_MERFISH_VolMean_Residue.csv"
# MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/MERFISH_v3_WB/"
# OutDir="/home/jw3514/Work/CellType_Psy/dat/Z2.Split/MouseSTR_MERFISH_VolMean_ResidueZ2.Z2/"
# OutFil="/home/jw3514/Work/CellType_Psy/dat/ResidueBiasMat/MouseSTR_MERFISH_VolMean_ResidueZ2.csv"
# run_z2_calculation "$ExpMat" "$MatchDir" "$OutDir" "$OutFil"


#ExpMat="/home/jw3514/Work/CellType_Psy/dat/ExpSpecific/MouseCT.NonAnchor.Residue.csv"
#OutDir="../dat/Z2.Split/MouseCT_ClusterV3_Residue_Z2_ISHMatch/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/dat/ExpSpecific/MouseCT.NonAnchor.Residue.csv"
#OutDir="../dat/Z2.Split/MouseCT_ClusterV3_Residue_Z2_MERFISHMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/MERFISH_v3_WB/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_MIT_CellMean_UMI_MF_Match_Z2/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/MERFISH_v3_WB/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_MIT_VolMean_UMI_MF_Match_Z2/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/MERFISH_v3_WB/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_V3_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/ClusterV3_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_V3_Z1Mat.clip3.csv"
#OutDir="../dat/Z2.Split/ClusterV3_UMI_ISHMatch_z1cplip3_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_MIT_CellMean_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 10 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_MIT_VolMean_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 10 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_Allen_CellMean_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_NEU_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_Allen_NEU_Mean_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_Allen_NEU_Vol_Mean_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Vol_Mean_Z1Mat.clip.csv"
#OutDir="../dat/Z2.Split/MERFISH_Allen_VolMean_UMI_ISHMatch_Z2/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &
#

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.csv"

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_V2_ExpMat.csv"
#OutDir="../dat/Z2.Split/MouseCT_Subclass_V2_Zmatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_V3_ExpMat.csv"
#OutDir="../dat/Z2.Split/MouseCT_Subclass_V3_Zmatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/cluster_V3_ExpMat.csv"
#OutDir="../dat/Z2.Split/MouseCT_Cluster_V3_Zmatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/JW_ExpMat.LogMean.0418.csv"
#OutDir="../dat/Z2.Split/MouseISH_ISHMatch_Zmatch/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Cell_Mean_DF.csv"
#OutDir="../dat/Z2.Split/MERFISH_Zhuang_CB_CellMean_V3_SCMatch_Zmatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Vol_Mean_DF.csv"
#OutDir="../dat/Z2.Split/MERFISH_Zhuang_CB_VolMean_V3_SCMatch_Zmatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
##nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.csv"

#
#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Cell_Mean_DF.Z1.csv"
#OutDir="../dat/Z2.Split/MERFISH_Zhuang_CB_CellMean_V3_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Vol_Mean_DF.Z1.csv"
#OutDir="../dat/Z2.Split/MERFISH_Zhuang_CB_VolMean_V3_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 15 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.csv"
#OutDir="../dat/Z2.Split/SC_STR_CellMean_V3_Zmatch_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir

#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Median/STR_Cell_Mean_DF.Z1.csv"
#OutDir="../dat/Z2.Split/SC_STR_median_CellMean_V3_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Median/STR_Vol_Mean_DF.Z1.csv"
#OutDir="../dat/Z2.Split/SC_STR_median_VolMean_V3_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.csv"
#OutDir="../dat/Z2.Split/SC_STR_CellMean_V3_Zmatch_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Vol_Mean_DF.csv"
#OutDir="../dat/Z2.Split/SC_STR_VolMean_V3_Zmatch_SCMatch/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#nohup parallel -j 20 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End) &

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats_Median/Subclass_V3_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Subclass_median_V3/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#parallel -j 30 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.Z1.csv"
#OutDir="../dat/Z2.Split/SC_STR_CellMean_AllenRegionMatch/"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#mkdir -p $OutDir
#parallel -j 30 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.Z1.csv"
#OutDir="./dat/Z2.Split/SC_STR_CellMean/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#parallel -j 30 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Vol_Mean_DF.Z1.csv"
#OutDir="../dat/Z2.Split/SC_STR_VolMean/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#parallel -j 30 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Z1Mat.subclass.HumanEntrez.csv"
#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpMat.subclass.HumanEntrez.csv"
#MatchDir="/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp/"
#OutDir="../dat/Z2.Split/subclass_AllenRegionMatch/"
#OutDir="../dat/Z2.Split/subclass_ABC_TotalExp_Match_Exp/"

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpMat.cluster.HumanEntrez.csv"
#OutDir="../dat/Z2.Split/cluster_ABC_TotalExp_Match_Exp/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)



#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/Z1Mat.cluster.HumanEntrez.csv"
#OutDir="../dat/Z2.Split/cluster_ABC_TotalExp_Match/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_V3_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Subclass_V3/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_V2_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Subclass_V2/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV2"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Cluster_V3_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Cluster_V3/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Cluster_V2_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Cluster_V2/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV2"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Cluster_CB_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Cluster_CB/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_CB_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Subclass_CB/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_CB_qn_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Subclass_CB_qn/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Subclass_CB_qn_Z1Mat_qn.csv"
#OutDir="../dat/Z2.Split/Subclass_CB_qn_qn/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Fusion_Exp_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Fusion_CB/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)


#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats/Fusion_Exp.csv"
#OutDir="../dat/Z2.Split/Fusion_Zmatch_v3/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_10xV3/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/test/Mouse.MNSP.csv"
#OutDir="../dat/Z2.Split/Test.MNSP.2/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)

#ExpMat="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats_Median/Fusion_Z1Mat.csv"
#OutDir="../dat/Z2.Split/Median_Fusion/"
#MatchDir="/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MatchGenes/ABC_TotalExp_CB/"
#mkdir -p $OutDir
#parallel -j 40 bash run_Z2.sh {} $Step $ExpMat $MatchDir $OutDir ::: $(seq $Start $Step $End)
