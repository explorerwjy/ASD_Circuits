from ASD_Circuits import *

def SimulateDenovo():
    HGNC = pd.read_csv("/Users/jiayao/Work/Resources/protein-coding_gene.txt", delimiter="\t")
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    allen_mouse_genes = loadgenelist("../dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")

    BGMR = pd.read_csv("/Users/jiayao/Work/Resources/MutationRate_20170711_rate.txt", delimiter="\t")
    BGMR["Entrez"] = [int(GeneSymbol2Entrez.get(x, -1)) for x in BGMR["GeneName"].values]
    BGMR = BGMR[BGMR["Entrez"].isin(allen_mouse_genes)]
    BGMR.index=BGMR["Entrez"].values

    N_simulations = 1000
    N_indv = 16877
    #N_simulations = 20
    #N_indv = 100

    DAT = []
    for i in range(N_simulations):
        LGDs1 = np.random.rand(N_indv, BGMR.shape[0])
        LGDs2 = np.random.rand(N_indv, BGMR.shape[0])
        Dmis1 = np.random.rand(N_indv, BGMR.shape[0])
        Dmis2 = np.random.rand(N_indv, BGMR.shape[0])
        LGD_idx = np.array([])
        Dmis_idx = np.array([])
        for j in range(N_indv):
            j_lgd_1 = np.where(BGMR["p_LGD"].values > LGDs1[j, :])
            j_lgd_2 = np.where(BGMR["p_LGD"].values > LGDs2[j, :])
            j_dmis_1 = np.where(BGMR["prevel_0.5"].values > Dmis1[j, :])
            j_dmis_2 = np.where(BGMR["prevel_0.5"].values > Dmis2[j, :])
            LGD_idx = np.append(LGD_idx, j_lgd_1)
            LGD_idx = np.append(LGD_idx, j_lgd_2)
            Dmis_idx = np.append(Dmis_idx, j_dmis_1)
            Dmis_idx = np.append(Dmis_idx, j_dmis_2)
            k,v = np.unique([BGMR.index[k] for k in LGD_idx], return_counts=True)
            Entrez_LGD = dict(zip(k, v))
            k,v = np.unique([BGMR.index[k] for k in Dmis_idx], return_counts=True)
            Entrez_Dmis = dict(zip(k, v))
            dat = {"LGD": Entrez_LGD, "Dmis": Entrez_Dmis}
        DAT.append(dat)

    keys = []
    for dat in DAT:
        keys.extend(list(dat["LGD"].keys()) + list(dat["Dmis"].keys()))
    keys = list(set(keys))

    dat_df = []
    for dat in DAT:
        row_df = []
        for k in keys:
            lgd = dat["LGD"].get(k, 0)
            dmis = dat["Dmis"].get(k, 0)
            row_df.append("%d,%d"%(lgd,dmis))
        dat_df.append(row_df)
    df = pd.DataFrame(dat_df, columns=keys) 
    df["index"] = df.index
    df = df.set_index('index')
    df.to_csv("dat/meta_denovo_simulations.csv")

def get_match_df(glist):
    dat = []
    _dir = "dat/Match_Volume_W/"
    _index = np.empty(0)
    no_g = []
    for g in glist: 
        try:
            tmp = loadgenelist(_dir + str(g) + ".txt.gz")
        except:
            no_g.append(g)
        dat.append(tmp)
        _index = np.append(_index, g)
    Sim_Denovo_Matching = pd.DataFrame(data=dat, index=_index)
    return Sim_Denovo_Matching

def Sim_DNV_ExpressionLevel():
    ExpMat = pd.read_csv("dat/Jon_data/energy-exp_level.csv", index_col="ROW")
    denovo_simulations = pd.read_csv("dat/meta_denovo_simulations.csv", index_col = "index")
    tmp_match_df = get_match_df(denovo_simulations.columns.values)
    for i, row in denovo_simulations.iterrows():
        weights = sim_denovo_row2gweight(row)
        tmp_match_df = get_match_df(weights.keys())
        sim_explevel = ExpAVGWithExpMatch(ExpMat, weights, tmp_match_df, csv_fil = "dat/bias/DNV_SM_ExpLevel/dnv_sm.{}.csv".format(i))

def Sim_DNV_AvgZ():
    ExpMat = pd.read_csv("../dat/allen-mouse-exp/energy-zscore-conn-model.csv", index_col="ROW")
    denovo_simulations = pd.read_csv("dat/meta_denovo_simulations.csv", index_col = "index")
    for i, row in denovo_simulations.iterrows():
        weights = sim_denovo_row2gweight(row)
        sim_avgz = AvgSTRZ_Weighted(ExpMat, weights, Method=1, csv_fil = "dat/bias/DNV_SM_AvgZ/dnv_sm.{}.csv".format(i))


def main():
    #SimulateDenovo()
    #Sim_DNV_ExpressionLevel()
    Sim_DNV_AvgZ()

main()
