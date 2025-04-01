# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_Cal_AvgExp_Cluster.py
# ========================================================================================================

import argparse
import os
import pandas as pd
import numpy as np
import anndata
import time
import matplotlib.pyplot as plt
import json
import requests
import pickle

def SaveDict(Dict, fname):
    with open(fname, 'wb') as hand:
        pickle.dump(Dict, hand)
    return

def LoadDict(fname):
    with open(fname, 'rb') as hand:
        b = pickle.load(hand)
        return b

def AggregateExpMat2Cluster(adata, cell_extended, label, name):
    # Class
    groupnames = cell_extended[label].unique()
    print(len(groupnames),"uniq groups")
    group2barcode = {}
    for groupname in groupnames:
        tmpdf = cell_extended[cell_extended[label]==groupname]
        group2barcode[groupname] = tmpdf["cell_barcode"].values
    dat = []
    group_names = []
    CellType2NCell = {}
    for group, cells in group2barcode.items():
        print(group, len(cells))
        cells = group2barcode[group]
        submat = adata[adata.obs["cell_barcode"].isin(cells)]
        if submat.shape[0] == 0: # No cell barcode in this cell type
            continue   
        CellType2NCell[group] = submat.shape[0]
        test = submat.to_df()
        #mean = test.mean(axis=0)
        mean = test.median(axis=0, skipna=True)
        group_names.append(group)
        dat.append(mean.values)
    dat = np.array(dat).transpose()
    df = pd.DataFrame(data=dat, index=adata.var.index.values, columns=group_names)
    #df.to_csv("../dat/AggregatedExpMat/ExpMat.{}.{}.csv".format(name, label))
    #SaveDict(CellType2NCell, "../dat/AggregatedExpMat/NumCells.{}.{}.csv".format(name, label))
    df.to_csv("../dat/AggregatedExpMat_Medium/ExpMat.{}.{}.csv".format(name, label))
    SaveDict(CellType2NCell, "../dat/AggregatedExpMat_Medium/NumCells.{}.{}.csv".format(name, label))
    return

class script_Cal_AvgExp_Cluster:
    def __init__(self, args):
        cellmeta_name = args.CellMeta.strip()
        self.adata = anndata.read_h5ad(args.InpFil, backed='r')
        print(cellmeta_name)
        self.cell_meta = pd.read_csv(cellmeta_name)
        self.Name = args.Name
        self.Label = args.Label
    def run(self):
        AggregateExpMat2Cluster(self.adata, self.cell_meta, self.Label, self.Name)

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InpFil', type=str, required=True, help='H5ad File of individual cell expression')
    parser.add_argument('-n', '--Name', type=str, help='Name of the dataset')
    parser.add_argument('-l', '--Label', type=str, help='Level of the Cluster, must be one of [class, subclass, supertype, cluster]')
    parser.add_argument('-c', '--CellMeta', type=str, required=True, help='Cell Annotation Metadata of the dataset')
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = script_Cal_AvgExp_Cluster(args)
    ins.run()
    return


if __name__ == '__main__':
    main()
