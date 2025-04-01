import os
import pandas as pd
import numpy as np
import anndata
import time
import matplotlib.pyplot as plt
import json
import requests
import pickle
import gzip as gz

import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src/')
from CellType_PSY import *

import multiprocessing
from multiprocessing import Pool


def worker(STR, MERFISH_STRAnn, MERFISH_CELL_DF):
    print("Processing ", STR)
    STR_MERFISH_DF = MERFISH_STRAnn[MERFISH_STRAnn["ISH_STR"]==STR]
    STR_Cells = MERFISH_CELL_DF[MERFISH_CELL_DF.obs.index.isin(STR_MERFISH_DF.index.values)]
    #tmpdf = STR_Cells.to_df()
    tmpdf = STR_Cells

    dat = []
    for g in tmpdf.columns.values:
        gmean = tmpdf[g].mean()
        dat.append(gmean)
    # dat 1 X NGene array of EXP of that STR
    print("                               Finish Processing", STR)
    return STR, dat

def MakeExpMat(results, genes):
    STRs = []
    dat_cell_avg = []
    dat_vol_avg = []
    for (STR, dat) in results:
        STRs.append(STR)
        dat_cell_avg.append(cell_avg_exp)
        dat_vol_avg.append(vol_avg_exp)
    STR_Cell_Mean_DF = pd.DataFrame(data=dat_cell_avg, index=STRs, columns=genes)
    STR_Vol_Mean_DF = pd.DataFrame(data=dat_vol_avg, index=STRs, columns=genes)
    STR_Cell_Mean_DF = STR_Cell_Mean_DF.transpose()
    STR_Vol_Mean_DF = STR_Vol_Mean_DF.transpose()
    STR_Cell_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_MarkerG/STR_Ce    ll_Mean_DF.csv")
    STR_Vol_Mean_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH_MarketG/STR_Vol    _Mean_DF.csv")

def main():
    MERFISH_Fil = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/abc_download_root/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad"
    MERFISH_SectionDir = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/abc_download_root/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad"

    print("Reading files")
    MERFISH_CELL_DF = anndata.read_h5ad(MERFISH_Fil, backed='r')
    #MERFISH_STRAnn = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/MERFISH.ISH_Annot.csv", index_col=0)
    MERFISH_STRAnn = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/MERFISH/Sample.csv", index_col=0)
    STRs = list(set(MERFISH_STRAnn["ISH_STR"]))

    MERFISH_CELL_DF = MERFISH_CELL_DF.to_df()
    

    print("Start processing...")
    pool = multiprocessing.Pool(processes=20)
    results = pool.starmap(worker, [(STR, MERFISH_STRAnn, MERFISH_CELL_DF) for STR in STRs])
    pool.close()
    pool.join()
    MakeExpMat(results, Cluster_V3_Exp.index.values)
    return

if __name__=='__main__':
    main()

