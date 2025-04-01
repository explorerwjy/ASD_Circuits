#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# script_STR_SC_ExpMat.py
#========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
from CellType_PSY import *
import multiprocessing
from multiprocessing import Pool


Non_Neuron_Class = ["30 Astro-Epen","31 OPC-Oligo","32 OEC","33 Vascular","34 Immune"]
    
def STR_SC_Exp(STR, Cluster_V3_Exp, CCF_DF, MERFISH):
    #MERFISH_STR = pd.read_csv("{}/Merfish-{}.csv".format(DIR, STR), index_col=0)
    MERFISH_STR = MERFISH[MERFISH["ISH_STR"]==STR]
    print("Processing {}; No.Cells:{}".format(STR, MERFISH_STR.shape[0]))
    STR_Volumns = CCF_DF.loc[STR, "total_voxel_counts (10 um)"]
    str_cell_sum_exp = np.zeros(Cluster_V3_Exp.shape[0])
    str_neuro_sum_exp = np.zeros(Cluster_V3_Exp.shape[0])
    str_neuro_count = 0
    for i, row in MERFISH_STR.iterrows():
        cluster = row["cluster"]
        cell_class = row["class"]
        if cluster in Cluster_V3_Exp.columns.values:
            cell_exp = Cluster_V3_Exp[cluster].values
            cell_exp = np.nan_to_num(cell_exp, nan=0)
            str_cell_sum_exp += cell_exp
            if cell_class not in Non_Neuron_Class:
                str_neuro_sum_exp += cell_exp
                str_neuro_count += 1
    if MERFISH_STR.shape[0] == 0:
        cell_avg_exp = str_cell_sum_exp
        neuro_avg_exp = str_neuro_sum_exp
    else:
        cell_avg_exp = str_cell_sum_exp / MERFISH_STR.shape[0]
        neuro_avg_exp = str_neuro_sum_exp / str_neuro_count
    vol_avg_exp = str_cell_sum_exp / STR_Volumns
    neuro_vol_avg_exp = str_neuro_sum_exp / STR_Volumns
    print("                        Finish" ,STR)
    return (STR, cell_avg_exp, vol_avg_exp, neuro_avg_exp, neuro_vol_avg_exp)
    
def MakeExpMat(results, genes):
    STRs = []
    dat_cell_avg = []
    dat_vol_avg = []
    dat_neuro_avg = []
    dat_neuro_vol_avg = []
    for (STR, cell_avg_exp, vol_avg_exp, neuro_avg_exp, neuro_vol_avg_exp) in results:
        STRs.append(STR)
        dat_cell_avg.append(cell_avg_exp)
        dat_vol_avg.append(vol_avg_exp)
        dat_neuro_avg.append(neuro_avg_exp)
        dat_neuro_vol_avg.append(neuro_vol_avg_exp)
    STR_Cell_Mean_DF = pd.DataFrame(data=dat_cell_avg, index=STRs, columns=genes)
    STR_Vol_Mean_DF = pd.DataFrame(data=dat_vol_avg, index=STRs, columns=genes)
    STR_NEU_Mean_DF = pd.DataFrame(data=dat_neuro_avg, index=STRs, columns=genes)
    STR_NEU_Vol_Mean_DF = pd.DataFrame(data=dat_neuro_vol_avg, index=STRs, columns=genes)
    STR_Cell_Mean_DF = STR_Cell_Mean_DF.transpose()
    STR_Vol_Mean_DF = STR_Vol_Mean_DF.transpose()
    STR_NEU_Mean_DF = STR_NEU_Mean_DF.transpose()
    STR_NEU_Vol_Mean_DF = STR_NEU_Vol_Mean_DF.transpose()
    STR_Cell_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Cell_Mean_DF.UMI.csv")
    STR_Vol_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_Vol_Mean_DF.UMI.csv")
    STR_NEU_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_NEU_Mean_DF.UMI.csv")
    STR_NEU_Vol_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/STR_NEU_Vol_Mean_DF.UMI.csv")
    #STR_Cell_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Cell_Mean_DF.UMI.csv")
    #STR_Vol_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_Vol_Mean_DF.UMI.csv")
    #STR_NEU_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/STR_NEU_Mean_DF.UMI.csv")

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help = '')
    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    CCF_DF = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/CCF_V3_ISH_MERFISH.csv", index_col="CleanName")
    STRs = CCF_DF.index.values
    Cluster_V3_Exp = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_MeanLogUMI.csv", index_col=0)
    #MERFISH = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/Sample.csv", index_col=0)
    MERFISH = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/MERFISH.ISH_Annot.clean.csv", index_col=0)
    #MERFISH = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_Zhuang/MERFISH_Zhuang_1-4_Combine.STR.Annot.csv", index_col=0)
    pool = multiprocessing.Pool(processes=20)
    results = pool.starmap(STR_SC_Exp, [(STR, Cluster_V3_Exp, CCF_DF, MERFISH) for STR in STRs])
    pool.close()
    pool.join()
    MakeExpMat(results, Cluster_V3_Exp.index.values)
    return

if __name__=='__main__':
    main()
