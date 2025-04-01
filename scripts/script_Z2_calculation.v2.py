#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_Z2_calculation.py
# Calculate Z2 bias for all genes
# ========================================================================================================

import argparse
import sys
import multiprocessing
import pandas as pd
import numpy as np
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src/')
from ASD_Circuits import *


def Z2_calculation(start_index, z1_mat, step, match_dir, out_dir):
    entrez2idx = dict(zip(np.arange(len(z1_mat.index.values)), z1_mat.index.values))
    STRs = z1_mat.columns.values
    dat = []
    index = []
    
    for idx in np.arange(start_index, start_index + step, 1):
        print(idx)
        try:
            entrez = entrez2idx[idx]
        except:
            continue
            
        try:
            match_genes = loadgenelist(f"{match_dir}/{entrez}.csv", toint=True)
            match_genes = [int(g) for g in match_genes if g in z1_mat.index.values]
            match_genes = match_genes[:500]
        except:
            print(entrez, "Match File not found")
            continue
            
        index.append(entrez)
        row_dat = []
        
        for STR in STRs:
            z_gene = z1_mat.loc[entrez, STR]
            if pd.isna(z_gene):
                row_dat.append(np.nan)
                continue
                
            z_matches = [z1_mat.loc[g, STR] for g in match_genes]
            z_matches = [x for x in z_matches if not pd.isna(x)]
            
            if len(z_matches) == 0:
                row_dat.append(np.nan)
                continue
                
            b = (z_gene - np.mean(z_matches)) / np.std(z_matches)
            row_dat.append(b)
            
        dat.append(row_dat)
        
    index = np.array(index)
    DF = pd.DataFrame(data=dat, index=index, columns=STRs)
    DF.to_csv(f"{out_dir}/{start_index}-{start_index+step}.z2.csv")


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
    parser.add_argument('-p', '--process', type=int, default=20, help='number of parallel processes')
    args = parser.parse_args()
    return args


def main():
    args = GetOptions()
    
    # Read Z1 matrix
    z1_mat = pd.read_csv(args.input, index_col=0)
    z1_mat.index = [int(x) for x in z1_mat.index.values]
    
    # Create pool and run parallel processes
    pool = multiprocessing.Pool(processes=args.process)
    
    # Generate indices for parallel processing
    indices = range(args.start, args.start + args.step * args.process, args.step)
    
    # Run parallel processes
    results = pool.starmap(Z2_calculation, 
                         [(idx, z1_mat, args.step, args.match, args.outdir) for idx in indices])
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
