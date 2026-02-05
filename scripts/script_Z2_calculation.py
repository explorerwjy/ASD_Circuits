#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_Z2_calculation.py
# Calculate Z2 bias for all genes
# ========================================================================================================

import argparse
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ASD_Circuits import *

class script_Z2_calculation:
    def __init__(self, args):
        self.z1_mat = pd.read_csv(args.input, index_col=0)
        self.z1_mat.index = [int(x) for x in self.z1_mat.index.values]
        self.start = args.start
        self.step = args.step
        self.matchDir = args.match
        self.outDir = args.outdir

    def run(self):
        entrez2idx = dict(
                zip(np.arange(len(self.z1_mat.index.values)), self.z1_mat.index.values))
        STRs = self.z1_mat.columns.values
        dat = []
        index = []
        for idx in np.arange(self.start, self.start + self.step, 1):
            print(idx)
            try:
                entrez = entrez2idx[idx]
            except:
                continue
            try:
                match_genes = loadgenelist(
                        self.matchDir+"/{}.csv".format(entrez), toint=True)
                match_genes = [
                        int(g) for g in match_genes if g in self.z1_mat.index.values]
                match_genes = match_genes[:1000]
            except:
                print(entrez, "Match Fil not find")
                continue
            index.append(entrez)
            row_dat = []
            #print(match_genes)
            for STR in STRs:
                z2_str = self.Z2_Gene_STR(entrez, STR, match_genes)
                row_dat.append(z2_str)
            dat.append(row_dat)
        index = np.array(index)
        DF = pd.DataFrame(data=dat, index=index, columns=STRs)
        DF.to_csv(
                "{}/{}-{}.z2.csv".format(self.outDir, self.start, self.start+self.step))

    def Z2_Gene_STR(self, gene, STR, match_genes):
        z_gene = self.z1_mat.loc[gene, STR]
        #print(gene, STR,  z_gene)
        if z_gene != z_gene:
            return np.nan
        z_matches = [self.z1_mat.loc[g, STR] for g in match_genes]
        #print(z_matches)
        #checkerr(self.z1_mat, STR, match_genes)
        z_matches = [x for x in z_matches if x == x]
        b = (z_gene - np.mean(z_matches)) / np.std(z_matches)
        return b

def checkerr(df, STR, match_genes):
    for g in match_genes:
        z = df.loc[g, STR]
        print(g, z.shape)

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
            type=str, help='expression spec (Z1) matrix')
    parser.add_argument('-s', '--start',  type=int,
            help='start index of genes')
    parser.add_argument('--step',  type=int, help='Number of genes each step')
    parser.add_argument('-m', '--match', type=str,
            help='Directory contains matched genes for all mouse gene')
    parser.add_argument('-o', '--outdir', type=str,
            help='Directory output')
    args = parser.parse_args()

    return args

def main():
    args = GetOptions()
    ins = script_Z2_calculation(args)
    ins.run()
    return


if __name__ == '__main__':
    main()
