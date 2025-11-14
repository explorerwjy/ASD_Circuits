# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_run_ctrl_sim.v3.py
# Run control simutations for expression biases (Cell Type; Structures)
# ========================================================================================================

import argparse
import sys
import os
import yaml

# Load config to get project directory
def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
ProjDIR = config["ProjDIR"]
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *
import time
import json
import pickle
import multiprocessing
from multiprocessing import Pool

###########################################################################
## Human Cell Type CTRL Generation
###########################################################################

def BiasCal_SingleCtrl_HumanCT(Random_GW_DF, SpecMat, idx):
    GW = dict(zip(Random_GW_DF[idx].values, Random_GW_DF["GeneWeight"].values))
    Ctrl_BiasDF = HumanCT_AvgZ_Weighted(SpecMat, GW)
    return Ctrl_BiasDF

def CtrlBiasCal_HumanCT(Random_GW_Fil, SpecMat, outfile, n_processes=20): 
    print("HumanCT Null Bias Simulation -", Random_GW_Fil)

    Random_GW_DF = pd.read_csv(Random_GW_Fil, index_col=0) # index is real gene,first column is gene weights, rest are random gene entrez
    Task_idx = Random_GW_DF.columns.values[1:]
    #print(Task_idx)

    if SpecMat.endswith('.parquet'):
        SpecMat = pd.read_parquet(SpecMat)
    else:
        SpecMat = pd.read_csv(SpecMat, index_col=0)
    SpecMat.columns = SpecMat.columns.astype(str)

    pool = multiprocessing.Pool(processes=n_processes)
    result_dfs = pool.starmap(BiasCal_SingleCtrl_HumanCT, [(Random_GW_DF, SpecMat, idx) for idx in Task_idx])

    pool.close()
    pool.join()
    print("end multiprocessing, merging results")
    str_ids = sorted(result_dfs[0].index.values)
    effects_matrix = np.stack([df.loc[str_ids, "EFFECT"].values for df in result_dfs], axis=0)
    effects_df = pd.DataFrame(effects_matrix.T, index=str_ids, columns=[str(i) for i in range(effects_matrix.shape[0])])
    effects_df.to_parquet(outfile)
    return
    

###########################################################################
## Mouse Cell Type CTRL Generation
###########################################################################

# New vectorized functions for pipeline
def BiasCal_SingleCtrl_MouseCT(Random_GW_DF, SpecMat, idx):
    GW = dict(zip(Random_GW_DF[idx].values, Random_GW_DF["GeneWeight"].values))
    BiasDF = MouseCT_AvgZ_Weighted(SpecMat, GW)
    return BiasDF

# New vectorized function for pipeline
def CtrlBiasCal_MouseCT(Random_GW_Fil, SpecMat, outfile, n_processes=20, DN=False): 
    print("MouseCT Null Bias Simulation -", Random_GW_Fil)
    Random_GW_DF = pd.read_csv(Random_GW_Fil, index_col=0) # index is real gene,first column is gene weights, rest are random gene entrez
    Task_idx = Random_GW_DF.columns.values[1:]
    #print(Task_idx)
    if SpecMat.endswith('.parquet'):
        SpecMat = pd.read_parquet(SpecMat)
    else:
        SpecMat = pd.read_csv(SpecMat, index_col=0)
    SpecMat.columns = SpecMat.columns.astype(str)
    pool = multiprocessing.Pool(processes=n_processes)

    result_dfs = pool.starmap(BiasCal_SingleCtrl_MouseCT, [(Random_GW_DF, SpecMat, idx) for idx in Task_idx])
    pool.close()
    pool.join()
    print("end multiprocessing, merging results")
    ct_ids = sorted(result_dfs[0].index.values)
    effects_matrix = np.stack([df.loc[ct_ids, "EFFECT"].values for df in result_dfs], axis=0)
    effects_df = pd.DataFrame(effects_matrix.T, index=ct_ids, columns=[str(i) for i in range(effects_matrix.shape[0])])
    effects_df.to_parquet(outfile)
    return

###########################################################################
## Mouse Structure CTRL Generation
###########################################################################
def BiasCal_SingleCtrl(GW_Fil, SpecMat, outDir):
    GW = Fil2Dict(GW_Fil)
    idx = GW_Fil.split("/")[-1].split(".")[2]
    Ctrl_BiasDF = MouseSTR_AvgZ_Weighted(SpecMat, GW, csv_fil="{}/cont.bias.{}.csv".format(outDir, idx))
    return

def BiasCal_SingleCtrl_MouseSTR(Random_GW_DF, SpecMat, idx):
    #GW = {gene: Random_GW_DF.loc[gene, idx] for gene in Random_GW_DF.index.values if gene in SpecMat.index.values}
    GW = dict(zip(Random_GW_DF[idx].values, Random_GW_DF["GeneWeight"].values))
    BiasDF = MouseSTR_AvgZ_Weighted(SpecMat, GW)
    return BiasDF

def CtrlBiasCal(GW_Dir, SpecMat, outDir, n_processes=20): 
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    Ctrl_GW_Fils = []
    for root, dirs, file_names in os.walk(GW_Dir):
        for file_name in file_names:
            Ctrl_GW_Fils.append(os.path.join(root, file_name))
    #print(Ctrl_GW_Fils)
    #print(len(Ctrl_GW_Fils))
    if SpecMat.endswith('.parquet'):
        SpecMat = pd.read_parquet(SpecMat)
    else:
        SpecMat = pd.read_csv(SpecMat, index_col=0)
    pool = multiprocessing.Pool(processes=n_processes)
    results = pool.starmap(BiasCal_SingleCtrl, [(GW_Fil, SpecMat, outDir) for GW_Fil in Ctrl_GW_Fils])
    pool.close()
    pool.join()

def CtrlBiasCal_MouseSTR(Random_GW_Fil, SpecMat, outfile, n_processes=20): 
    print("MouseSTR Null Bias Simulation -", Random_GW_Fil)
    Random_GW_DF = pd.read_csv(Random_GW_Fil, index_col=0) # index is real gene,first column is gene weights, rest are random gene entrez
    Task_idx = Random_GW_DF.columns.values[1:]
    #print(Task_idx)
    if SpecMat.endswith('.parquet'):
        SpecMat = pd.read_parquet(SpecMat)
    else:
        SpecMat = pd.read_csv(SpecMat, index_col=0)
    SpecMat.columns = SpecMat.columns.astype(str)
    pool = multiprocessing.Pool(processes=n_processes)
    result_dfs = pool.starmap(BiasCal_SingleCtrl_MouseSTR, [(Random_GW_DF, SpecMat, idx) for idx in Task_idx])
    pool.close()
    pool.join()
    print("end multiprocessing, merging results")
    str_ids = sorted(result_dfs[0].index.values)
    effects_matrix = np.stack([df.loc[str_ids, "EFFECT"].values for df in result_dfs], axis=0)
    effects_df = pd.DataFrame(effects_matrix.T, index=str_ids, columns=[str(i) for i in range(effects_matrix.shape[0])])
    effects_df.to_parquet(outfile)
    return

###########################################################################
## Args and Main Functions
###########################################################################
def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='Mode of program 1: gene weight generateion; 2. bias calculaton')
    parser.add_argument('-o', '--outfile', type=str, help='Output file')
    parser.add_argument('--Ctrl_Genes_Fil', type=str, required=True, help="Filename of ctrl genes")
    parser.add_argument('--SpecMat', type=str, help="Filename of bias matrix")
    parser.add_argument('--n_processes', type=int, default=20, help="Filename of bias matrix")
    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    mode = args.mode
    if mode == 'bias':
        SpecMat = args.SpecMat
        Ctrl_Genes_Fil = args.Ctrl_Genes_Fil
        outfile = args.outfile
        n_processes = args.n_processes
        CtrlBiasCal(Ctrl_Genes_Fil, SpecMat, outfile, n_processes) # 
    if mode == 'human_ct_bias':
        SpecMat = args.SpecMat
        Ctrl_Genes_Fil = args.Ctrl_Genes_Fil
        outfile = args.outfile
        n_processes = args.n_processes
        CtrlBiasCal_HumanCT(Ctrl_Genes_Fil, SpecMat, outfile, n_processes)
    if mode == "mouse_ct_bias":
        SpecMat = args.SpecMat
        Ctrl_Genes_Fil = args.Ctrl_Genes_Fil
        outfile = args.outfile
        n_processes = args.n_processes
        CtrlBiasCal_MouseCT(Ctrl_Genes_Fil, SpecMat, outfile, n_processes)
    if mode == "mouse_str_bias":
        SpecMat = args.SpecMat
        Ctrl_Genes_Fil = args.Ctrl_Genes_Fil
        outfile = args.outfile
        n_processes = args.n_processes
        CtrlBiasCal_MouseSTR(Ctrl_Genes_Fil, SpecMat, outfile, n_processes)

    return

if __name__ == '__main__':
    main()
