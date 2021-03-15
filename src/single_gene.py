from ASD_Circuits import *

# Load Exp Zscore Matrix
ExpMat = "../dat/allen-mouse-exp/energy-conn-model.csv"
ExpMat = pd.read_csv(ExpMat, index_col="ROW")

ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-conn-model.csv"
#ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-neuronorm.csv"
ExpZscoreMat = pd.read_csv(ExpZscoreMat, index_col="ROW")
allen_mouse_genes = loadgenelist("/Users/jiayao/Work/ASD_Circuits/dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")

asd_asc_match_df = pd.read_csv("dat/asd_asc_exp_matches_1000.csv", index_col="GENE")

#df = ZscoreAVGWithExpMatch_SingleGene(ExpZscoreMat, asd_asc_match_df.index, asd_asc_match_df)
df = Zscore_SingleGene(ExpZscoreMat, asd_asc_match_df.index, asd_asc_match_df)
df.to_csv("dat/bias/asc.zmatch.single.gene.csv")
