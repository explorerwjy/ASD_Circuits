#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_Z2_calculation.py
# Calculate Z2 bias for all genes
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *
#import numpy as np
#import pandas as pd
#import csv


#ExpZscoreMatFil = "JW_Z1-Mat.ArithmeticMean.0422.csv"
#ExpZscoreMatFil = "/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/AllenMouse_z1_mat.0511.csv"
#MatchDir = "/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/ExpMatch_RootExp_uniform_kernal"

ExpZscoreMatFil = "/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/energy-zscore-conn-model.0920.neuronorm.csv"
MatchDir = "/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/" 

#ExpZscoreMatFil = "/home/jw3514/Work/ASD_Circuits/dat/Human.CT.Z1.csv"
#MatchDir = "/home/jw3514/Work/ASD_Circuits/dat/genes/HumanCellType_ExpMatch_RootExp_uniform_kernal"

#ExpZscoreMatFil = "/home/jw3514/Work/CellType_Psy/dat/Human.CT.AllCell.Z1.Entrez.csv"
#MatchDir = "/home/jw3514/Work/ASD_Circuits/dat/genes/BrainSpanMatch_uniform_kernal/"

#ExpZscoreMatFil = "../dat/allen-mouse-exp/AllenMouse_z1_mat.0511.csv"
#MatchDir = "/home/jw3514/Work/ASD_Circuits/dat/genes/BrainSpanMatch_uniform_kernal/"

class script_Z2_calculation:
    def __init__(self, args):
        self.z1_mat = pd.read_csv(args.input, index_col=0)
        self.start = args.start
        #self.end = args.end
        self.step = args.step
        self.matchDir = args.match
        #self.method = args.method

    def run(self):
        entrez2idx = dict(
                zip(np.arange(len(self.z1_mat.index.values)), self.z1_mat.index.values))
        STRs = self.z1_mat.columns.values
        dat = []
        index = []
        for idx in np.arange(self.start, self.start + self.step, 1):
            try:
                entrez = entrez2idx[idx]
            except:
                continue
            #try:
                #match_genes = loadgenelist(
                #    self.matchDir+"/{}.csv".format(entrez), toint=True)
            try:
                match_genes = loadgenelist(
                        self.matchDir+"/{}.csv".format(entrez), toint=False)
                match_genes = [int(g) for g in match_genes]
                match_genes = [
                        g for g in match_genes if g in self.z1_mat.index.values]
            except:
                #print(entrez, "Match Fil not find")
                continue
            index.append(entrez)
            row_dat = []
            #print(entrez, match_genes)
            for STR in STRs:
                z2_str = self.Z2_Gene_STR(entrez, STR, match_genes)
                row_dat.append(z2_str)
            dat.append(row_dat)
        index = np.array(index)
        DF = pd.DataFrame(data=dat, index=index, columns=STRs)
        #DF.to_csv("../dat/HumanCT.BrainSpanMatch.Z2scores/{}-{}.z2.csv".format(self.start, self.start+self.step))

        #DF.to_csv("../dat/HumanCT.ACT.BrainSpanMatch.Z2score/{}-{}.z2.csv".format(self.start, self.start+self.step))
        #DF.to_csv("../dat/AllenMouse.NeuroNorm.Z2score/{}-{}.z2.csv".format(self.start, self.start+self.step))
        DF.to_csv("../dat/AllenMouse.ZMatch.Z2/{}-{}.z2.csv".format(self.start, self.start+self.step))
    def Z2_Gene_STR(self, gene, STR, match_genes):
        z_gene = self.z1_mat.loc[gene, STR]
        if z_gene != z_gene:
            return np.nan
        z_matches = [self.z1_mat.loc[g, STR] for g in match_genes]
        z_matches = [x for x in z_matches if x == x]
        b = (z_gene - np.mean(z_matches)) / np.std(z_matches)
        return b


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=ExpZscoreMatFil,
            type=str, help='expression spec (Z1) matrix')
    parser.add_argument('-s', '--start',  type=int,
            help='start index of genes')
    parser.add_argument('--step',  type=int, help='Number of genes each step')
    #parser.add_argument('-e','--end',  type=int, help = 'end index of genes')
    #parser.add_argument('--method', type=str, default=Z2, help='Method')
    parser.add_argument('-m', '--match', type=str, default=MatchDir,
            help='Directory contains matched genes for all mouse gene')
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = script_Z2_calculation(args)
    ins.run()
    return


if __name__ == '__main__':
    main()
