#!/usr/bin/env python3
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_Z2_calculation.py
# Calculate Z2 bias for all genes
# ========================================================================================================

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ASD_Circuits import *
from tqdm import tqdm


class script_Z2_calculation:
    def __init__(self, args):
        self.z1_mat = pd.read_csv(args.input, index_col=0)
        self.z1_mat.index = self.z1_mat.index.astype(int)
        self.start = args.start
        self.step = args.step
        self.matchDir = Path(args.match)
        self.outDir = Path(args.outdir)

    def run(self):
        entrez2idx = dict(enumerate(self.z1_mat.index))
        STRs = self.z1_mat.columns
        data = []
        index = []

        for idx in tqdm(range(self.start, self.start + self.step)):
            try:
                entrez = entrez2idx[idx]
            except KeyError:
                continue

            try:
                match_genes = loadgenelist(self.matchDir / f"{entrez}.csv", toint=True)
                match_genes = [g for g in match_genes if g in self.z1_mat.index][:1000]
            except FileNotFoundError:
                print(f"{entrez}: Match File not found")
                continue

            index.append(entrez)
            row_dat = [self.Z2_Gene_STR(entrez, STR, match_genes) for STR in STRs]
            data.append(row_dat)

        DF = pd.DataFrame(data=data, index=index, columns=STRs)
        DF.to_csv(self.outDir / f"{self.start}-{self.start+self.step}.z2.csv")

    def Z2_Gene_STR(self, gene, STR, match_genes):
        z_gene = self.z1_mat.loc[gene, STR]
        if pd.isna(z_gene):
            return np.nan
        z_matches = self.z1_mat.loc[match_genes, STR].dropna()
        return (z_gene - z_matches.mean()) / z_matches.std() if not z_matches.empty else np.nan

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
            type=str, help='expression spec (Z1) matrix')
    parser.add_argument('-s', '--start', type=int,
            help='start index of genes')
    parser.add_argument('--step', type=int, help='Number of genes each step')
    parser.add_argument('-m', '--match', type=str,
            help='Directory contains matched genes for all mouse gene')
    parser.add_argument('-o', '--outdir', type=str,
            help='Directory output')
    return parser.parse_args()

def main():
    args = GetOptions()
    ins = script_Z2_calculation(args)
    ins.run()

if __name__ == '__main__':
    main()
