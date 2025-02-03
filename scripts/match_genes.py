import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *
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


def old_main():
    args = GetOptions()
    match_feature = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/match-features.csv", index_col="GENE")
    match_feature = match_feature.sort_values("EXP")
    match_feature["Rank"] = [1+x for x in range(match_feature.shape[0])]
    if args.bias == "specificity":
        #ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-conn-model.csv"
        ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-neuronorm.csv"
        ExpZscoreMat = pd.read_csv(ExpZscoreMat, index_col="ROW")
        ## Match gene set with exp-match genes
        gene_list = loadgenelist(args.input)
        if args.match == None:
            match_df = ExpressionMatchGeneSet(gene_list, match_feature)
        else:
            match_df = pd.read_csv(args.match, index_col="GENE")
        #eff_df = ZscoreAVGWithExpMatch(ExpZscoreMat, gene_list, match_df)
        gnomad_cons = pd.read_csv("/Users/jiayao/Work/Resources/gnomad.v2.1.1.lof_metrics.by_gene.txt", delimiter="\t")
        Gene2LoFZ = {}
        for i, row in gnomad_cons.iterrows():
            try:
                Gene2LoFZ[GeneSymbol2Entrez[row["gene"]]] = max(1, row["lof_z"] + 1)
            except:
                continue
        eff_df = ZscoreAVGWithExpMatch_Constraint(ExpZscoreMat, gene_list, match_df, Gene2LoFZ)

        eff_df = eff_df.sort_values("EFFECT", ascending=False)
        eff_df = eff_df.reset_index()
        eff_df["Rank"] = [x+1 for x in eff_df.index]
        eff_df.to_csv(args.output, index=False)

def Match_a_Gene(Gene_ID, MatchFeature, sample_size=10000, outfil = None, kernel="uniform", 
                 interval=0.05):
    gene_rank = MatchFeature.loc[Gene_ID, "Rank"]
    quantile = MatchFeature.loc[Gene_ID, "quantile"]
    #print(Gene_ID, gene_rank, quantile)
    min_quantile = max(0, quantile - interval)
    max_quantile = min(1, quantile + interval)
    Interval_genes = MatchFeature[(MatchFeature["quantile"]>=min_quantile)&
                               (MatchFeature["quantile"]<=max_quantile)&
                               (MatchFeature.index != Gene_ID)].index.values
    if kernel == "uniform":
        match_genes = np.random.choice(Interval_genes, size=sample_size, replace=True)
    elif kernel == "tricubic":
        print("To Be Implemented")
    if outfil:
        List2Fil(match_genes, outfil)

def main():
    #MatchFeature = pd.read_csv("../dat/genes/Match_Use_Root_Exp_Energy.csv",index_col=0)
    #MatchFeature = pd.read_csv("GeneMatchQuantile.csv",index_col=0)
    #MatchFeature = pd.read_csv("/home/jw3514/Work/ASD_Circuits/notebooks/HumanCellTypeGeneMatchQuantile.csv",index_col=0)
    #OutDir = "../dat/genes/ExpMatch_RootExp_uniform_kernal/"
    #OutDir = "../dat/genes/HumanCellType_ExpMatch_RootExp_uniform_kernal/"

    MatchFeature = pd.read_csv("../dat/Other/BrainSpanMatchQuantile.csv", index_col=0)
    OutDir = "../dat/genes/BrainSpanMatch_uniform_kernal/"
    for g, row in MatchFeature.iterrows():
        Match_a_Gene(g, MatchFeature, outfil="{}/{}.csv".format(OutDir, g))

if __name__=='__main__':
    main()

