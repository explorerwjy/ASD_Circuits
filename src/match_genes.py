from ASD_Circuits import *
import sys
import argparse

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help = 'Input genelist')
    parser.add_argument('-m', '--match', required=True, type=str, help = 'Input Matched genelist')
    parser.add_argument('-o', '--output', required=True, type=str, help = 'Output fil')
    parser.add_argument('-b', '--bias', default="specificity", type=str, help = "Bias Method [specificity] or [expression]")
    args = parser.parse_args()
    if args.output == None:
        args.output = args.input.split(".")[0] + ".snploc"
    return args


def main():
    args = GetOptions()
    match_feature = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/match-features.csv", index_col="GENE")
    match_feature = match_feature.sort_values("EXP")
    match_feature["Rank"] = [1+x for x in range(match_feature.shape[0])]
    if args.bias == "specificity":
        ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-conn-model.csv"
        ExpZscoreMat = pd.read_csv(ExpZscoreMat, index_col="ROW")
        ## Match gene set with exp-match genes
        gene_list = loadgenelist(args.input)
        if args.match == None:
            match_df = ExpressionMatchGeneSet(gene_list, match_feature)
        else:
            match_df = pd.read_csv(args.match, index_col="GENE")
        eff_df = ZscoreAVGWithExpMatch(ExpZscoreMat, gene_list, match_df)

        eff_df = eff_df.sort_values("EFFECT", ascending=False)
        eff_df = eff_df.reset_index()
        eff_df["Rank"] = [x+1 for x in eff_df.index]
        eff_df.to_csv(args.output, index=False)

if __name__=='__main__':
    main()

