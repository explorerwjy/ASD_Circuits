import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src/')
from CellType_PSY import *
import re
import json
import anndata
import pickle


#SaveDIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/Cluster_GeneXCell_UMI/"
SaveDIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/Cluster_GeneXCell_UMI_v2/"


def ExtractGeneXcellForCluster_v3(cluster, cell_extended, feature2Fil, ENSMUSG2HumanEntrez):
    cluster_meta = cell_extended[cell_extended["cluster"]==cluster]
    tmp_value_counts = cluster_meta["feature_matrix_label"].value_counts()
    CT_V3_features = [x for x in tmp_value_counts.index.values if "v3" in x]
    Cluster_Barcodes = cell_extended[cell_extended["cluster"]==cluster]["cell_barcode"].values
    print("CT: {} Ncells: {}".format(cluster, len(Cluster_Barcodes)))
    Cluster_GeneXCell_DFs = []
    for feature in CT_V3_features:
        feature_h5d = feature2Fil[feature]
        submat_cluster = feature_h5d[feature_h5d.obs["cell_barcode"].isin(Cluster_Barcodes)]
        submat_cluster = submat_cluster.to_df()
        submat_cluster = submat_cluster.transpose()
        submat_cluster = ExpMatConvertEntrez(submat_cluster, ENSMUSG2HumanEntrez)
        Cluster_GeneXCell_DFs.append(submat_cluster)
    cluster_clean_name = re.sub(r'\W+', '_', cluster)
    concatenated_df = pd.concat(Cluster_GeneXCell_DFs, axis=1)
    concatenated_df.to_csv("{}/{}.GeneXCell.csv".format(SaveDIR, cluster_clean_name))

def ExtractGeneXcellForCluster_v2(cluster, cell_extended, feature2Fil, ENSMUSG2HumanEntrez):
    cluster_meta = cell_extended[cell_extended["cluster"]==cluster]
    tmp_value_counts = cluster_meta["feature_matrix_label"].value_counts()
    CT_V2_features = [x for x in tmp_value_counts.index.values if "v2" in x]
    Cluster_Barcodes = cell_extended[cell_extended["cluster"]==cluster]["cell_barcode"].values
    print("CT: {} Ncells: {}".format(cluster, len(Cluster_Barcodes)))
    Cluster_GeneXCell_DFs = []
    for feature in CT_V2_features:
        feature_h5d = feature2Fil[feature]
        submat_cluster = feature_h5d[feature_h5d.obs["cell_barcode"].isin(Cluster_Barcodes)]
        submat_cluster = submat_cluster.to_df()
        submat_cluster = submat_cluster.transpose()
        submat_cluster = ExpMatConvertEntrez(submat_cluster, ENSMUSG2HumanEntrez)
        Cluster_GeneXCell_DFs.append(submat_cluster)
    cluster_clean_name = re.sub(r'\W+', '_', cluster)
    concatenated_df = pd.concat(Cluster_GeneXCell_DFs, axis=1)
    concatenated_df.to_csv("{}/{}.GeneXCell.csv".format(SaveDIR, cluster_clean_name))


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

def test(cell_extended, feature2Fil_10xV3, ENSMUSG2HumanEntrez):
    cluster = "2691 RE-Xi Nox4 Glut_1"
    ExtractGeneXcellForCluster_v3(cluster, cell_extended, feature2Fil_10xV3, ENSMUSG2HumanEntrez)

def run_all_cluster(ClusterAnn, cell_extended, feature2Fil, ENSMUSG2HumanEntrez):
    for cluster in ClusterAnn.index.values:
    #    ExtractGeneXcellForCluster_v2(cluster, cell_extended, feature2Fil, ENSMUSG2HumanEntrez)
        try:
            #ExtractGeneXcellForCluster_v3(cluster, cell_extended, feature2Fil, ENSMUSG2HumanEntrez)
            ExtractGeneXcellForCluster_v2(cluster, cell_extended, feature2Fil, ENSMUSG2HumanEntrez)
        except Exception as e:
            print("ERROR for cluster {}: {}".format(cluster, str(e)))

def main_v3():
    ClusterAnn = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label")
    cell_extended = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/cell_metadata_with_cluster_annotation.csv")
    ENSMUSG2HumanEntrez = json.load(open("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ENSMUSG2HumanEntrez.json", 'r'))
    feature_matrix_label = cell_extended["feature_matrix_label"].unique()
    feature_matrix_label_v3 = [x for x in feature_matrix_label if 'v3' in x]
    RAW_EXP_DIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/abc_download_root/expression_matrices/WMB-10Xv3/20230630/"
    feature2Fil_10xV3 = {}
    for feature in feature_matrix_label_v3:
        Adat = anndata.read_h5ad(RAW_EXP_DIR + "{}-raw.h5ad".format(feature), backed='r')
        feature2Fil_10xV3[feature] = Adat
    #test(cell_extended, feature2Fil_10xV3, ENSMUSG2HumanEntrez)
    run_all_cluster(ClusterAnn, cell_extended, feature2Fil_10xV3, ENSMUSG2HumanEntrez)

def main_v2():
    ClusterAnn = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label")
    cell_extended = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/cell_metadata_with_cluster_annotation.csv")
    ENSMUSG2HumanEntrez = json.load(open("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/ENSMUSG2HumanEntrez.json", 'r'))
    feature_matrix_label = cell_extended["feature_matrix_label"].unique()
    feature_matrix_label_v2 = [x for x in feature_matrix_label if 'v2' in x]
    #RAW_EXP_DIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/abc_download_root/expression_matrices/WMB-10Xv2/20230630/"
    RAW_EXP_DIR = "/mnt/data0/AllenMouseSC/abc_download_root/expression_matrices/WMB-10Xv2/20230630/"
    feature2Fil_10xV2 = {}
    for feature in feature_matrix_label_v2:
        Adat = anndata.read_h5ad(RAW_EXP_DIR + "{}-raw.h5ad".format(feature), backed='r')
        feature2Fil_10xV2[feature] = Adat
    #test(cell_extended, feature2Fil_10xV3, ENSMUSG2HumanEntrez)
    #print(feature2Fil_10xV2)
    run_all_cluster(ClusterAnn, cell_extended, feature2Fil_10xV2, ENSMUSG2HumanEntrez)

# main_v3()
main_v2()
