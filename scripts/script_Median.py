import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src/')
from CellType_PSY import *
import multiprocessing
from multiprocessing import Pool

def SaveDict(Dict, fname):
    with open(fname, 'wb') as hand:
        pickle.dump(Dict, hand)
    return

def LoadDict(fname):
    with open(fname, 'rb') as hand:
        b = pickle.load(hand)
        return b

def Workers(g, CellTypes, feature_matrix_label, Splited_Mats, NumCells):
    res = []
    for i, CT in enumerate(CellTypes):
        dat = []
        for feature in feature_matrix_label:
            SubExpMat = Splited_Mats[feature]
            CellNum = NumCells[feature] 
            if CT in CellNum:
                dat.extend([SubExpMat.loc[g, CT]]*CellNum[CT])
        if len(dat) > 0:
            median = np.median(dat)
        else:
            median = 0
        res.append(median)
    return (g, res)
        #CombineMat_v3.loc[g, CT] = median

def ExpMatConvertEntrez(CombineMat, ENSMUSG2HumanEntrez):
    dat_rows = []
    index = []
    for ensg, row in CombineMat.iterrows():
        if ensg in ENSMUSG2HumanEntrez:
            for entrez in ENSMUSG2HumanEntrez[ensg]:
                index.append(entrez)
                dat_rows.append(row)
    ExpDF_humanEntrez = pd.DataFrame(data=dat_rows, index=index, columns=CombineMat.columns.values)
    ExpDF_humanEntrezDedup = ExpDF_humanEntrez[~ExpDF_humanEntrez.index.duplicated(keep='first')]
    return ExpDF_humanEntrezDedup

def CollectResults(res, Genes, CellTypes, label):
    G_medians = {}
    for k,v in res:
        G_medians[k] = v
    dat = []
    for g in Genes:
        dat.append(G_medians[g])
    df = pd.DataFrame(data=dat, index=Genes, columns=CellTypes)
    df.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ExpressionMats_Median/test.{}_V3_ExpMat.csv".format(label))
    #ExpMatConvertEntrez(CombineMat_v3, ENSMUSG2HumanEntrez)
    #Subclass_V3_ExpMat.to_csv("dat/ExpressionMats_Median/{}_V3_ExpMat.csv".format(label))

def main():
    Split_Exp_Dir = "/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/AggregatedExpMat_Medium/"
    cell_extended = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/cell_metadata_with_cluster_annotation.csv")
    feature_matrix_label = cell_extended["feature_matrix_label"].unique()
    feature_matrix_label_v3 = [x for x in feature_matrix_label if 'v3' in x]
    #label = "cluster"
    label = "supertype"
    #label = "subclass"
    Splited_Mats_v3 = {}
    NumCells_v3 = {}
    for feature in feature_matrix_label_v3:
        SubExpMat = pd.read_csv("{}/ExpMat.{}.{}.csv".format(Split_Exp_Dir, feature, label), index_col=0)
        CellNum = LoadDict("{}/NumCells.{}.{}.csv".format(Split_Exp_Dir, feature, label))
        print(SubExpMat.shape)
        Splited_Mats_v3[feature] = SubExpMat
        NumCells_v3[feature] = CellNum

    CellTypes = cell_extended[label].unique()
    CellTypes.sort()
    Genes = Splited_Mats_v3[feature_matrix_label_v3[0]].index.values
    
    pool = multiprocessing.Pool(processes=20) 
    results = pool.starmap(Workers, [(G, CellTypes, feature_matrix_label_v3, Splited_Mats_v3, NumCells_v3) for G in Genes])
    CollectResults(results, Genes, CellTypes, label)

main()
