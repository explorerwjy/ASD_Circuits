#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_bootstrapping_mutations.py
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '../src')
from ASD_Circuits import *


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol, allen_mouse_genes = LoadGeneINFO()


class script_bootstrapping_mutations:
    def __init__(self, args):
        self.idx = args.idx
        self.Var = args.variant

    def run(self):
        ExpZscoreMat = "../dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv"
        ExpZscoreMat = pd.read_csv(ExpZscoreMat, index_col=0)
        #Mutation_n_Phenotype = pd.read_csv(self.Var)
        Mutation_n_Phenotype = pd.read_csv("../dat/Other/IQ_Mutation.Mar17.2023.csv")

        Mutation_n_Phenotype_perm = Mutation_n_Phenotype.copy(deep=True)
        Mutation_n_Phenotype_perm["Sex"] = np.random.permutation(
            Mutation_n_Phenotype_perm["Sex"].values)
        Male = Mutation_n_Phenotype_perm[Mutation_n_Phenotype_perm["Sex"]=="Male"]
        Female = Mutation_n_Phenotype_perm[Mutation_n_Phenotype_perm["Sex"]=="Female"]
        Male_gene2MutN = Mut2GeneDF(Male)
        Male_Spec = AvgSTRZ_Weighted(ExpZscoreMat, Male_gene2MutN,
                                       Method=1, csv_fil="../dat/Unionize_bias/Permutations/Gender_permut/Male.bias.perm.{}.csv".format(self.idx))
        Female_gene2MutN = Mut2GeneDF(Female)
        Female_Spec = AvgSTRZ_Weighted(ExpZscoreMat, Female_gene2MutN,
                                       Method=1, csv_fil="../dat/Unionize_bias/Permutations/Gender_permut/Female.bias.perm.{}.csv".format(self.idx))


def Mut2GeneDF(MutDF):
    genes = np.array(list(set(MutDF["Entrez"].values)))
    dat = []
    gene2MutN = {}
    for g in genes:
        Muts = MutDF[MutDF["Entrez"]==g]
        try:
            pLI = float(Muts["ExACpLI"].values[0])
        except:
            pLI = 0.0
        N_LGD, N_Mis, N_Dmis, N_Syn = CountMut(Muts)
        if pLI >= 0.5:
            gene2MutN[g] = N_LGD * 0.554 + N_Dmis * 0.333
        else:
            gene2MutN[g] = N_LGD * 0.138 + N_Dmis * 0.130
    return gene2MutN


def CountMut(DF):
    N_LGD, N_mis, N_Dmis, N_syn = 0, 0, 0, 0
    for i, row in DF.iterrows():
        GeneEff = row["GeneEff"].split(";")[0]
        if GeneEff in ["frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_gained", "stop_lost"]:
            N_LGD += 1
        elif GeneEff == "missense":
            N_mis += 1
            row["REVEL"] = row["REVEL"].split(";")[0]
            if row["REVEL"] != ".":
                if float(row["REVEL"]) > 0.5:
                    N_Dmis += 1
        elif GeneEff == "synonymous":
            N_syn += 1
    return N_LGD, N_mis, N_Dmis, N_syn


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--variant', type=str, default="dat/IQ_HC_muts.csv",
                        help='variant table used for permutation')
    parser.add_argument('-p', '--phenotype', type=str, default="dat/IQ_HC_Phenotype.csv",
                        help='phenotype table used for permutation')
    parser.add_argument('-m', '--match', type=str,
                        default="dat/Final_Spark_Meta.Matches.csv", help='Matched gene table for Z-Z')
    parser.add_argument('-i', '--idx', type=str, required=True, help='index')
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = script_bootstrapping_mutations(args)
    ins.run()

    return


if __name__ == '__main__':
    main()
