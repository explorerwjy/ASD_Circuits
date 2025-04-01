#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# script_STR_SC_ExpMat.py
#========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
from CellType_PSY import *
import re
import multiprocessing
from multiprocessing import Pool


def processCluster(cluster, DatDIR):
    try:
        print("Processing {}".format(cluster))
        cluster_clean_name = re.sub(r'\W+', '_', cluster)
        test_genexcell = pd.read_csv("{}/{}.GeneXCell.csv".format(DatDIR, cluster_clean_name), index_col=0)
        test_genexcell_log = np.log2(test_genexcell+1)
        Gene_Cluster_Mean = test_genexcell_log.mean(axis=1)
        Gene_Cluster_Total = test_genexcell_log.sum(axis=1)
        Total_UMI = test_genexcell_log.sum(axis=0)
        Gene_Cluster_Mean.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/SplitCTs/{}.csv".format(cluster_clean_name))
        return Gene_Cluster_Mean, Gene_Cluster_Total, Total_UMI
    except:
        print("ERROR {}".format(cluster))
        return np.zeros(17939), np.zeros(17939), []

def processClusterV2(cluster, DatDIR):
    try:
        print("Processing {}".format(cluster))
        cluster_clean_name = re.sub(r'\W+', '_', cluster)
        with open("{}/{}.GeneXCell.csv".format(DatDIR, cluster_clean_name)) as csvfile:
            gene_dat = []
            gene_index = []
            reader = csv.reader(csvfile)
            head = next(reader)
            for row in reader:
                gene_index.append(int(row[0]))
                log2UMI = np.log2(np.array([float(x) for x in row[1:]]) + 1)
                gene_dat.append(log2UMI.mean())
            Gene_Cluster_Mean = pd.Series(data=gene_dat, index=gene_index)
            Gene_Cluster_Mean.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/SplitCTs/{}.csv".format(cluster_clean_name))
    except:
        print("ERROR {}".format(cluster))

def CollectRes(clusters, results):
    Indv_cluster_means = []
    Indv_cluster_total_UMIs = []
    Gene_Total_Exp = np.zeros(len(results[0][1]))
    Total_N_Cells = 0
    for cluster, res in zip(clusters, results):
        if len(res[2]) == 0: # skip cluster dont have cells in v3
            continue
        gene_mean_logUMI = res[0]
        gene_mean_logUMI.name = cluster
        gene_total_logUMI = res[1]
        Gene_Total_Exp += gene_total_logUMI
        cell_depth = res[2].values
        Indv_cluster_means.append(gene_mean_logUMI)
        Indv_cluster_total_UMIs.append(cell_depth)
        Total_N_Cells += len(cell_depth)
    # Make and save cluster Exp Mat
    Cluster_Exp_DF = pd.concat(Indv_cluster_means, axis=1)
    Cluster_Exp_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/cluster_MeanLogUMI.csv")
    
    # Save Cell depth for each clusters
    with open("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/clusters_CellDepth.npy", 'wb') as f:
        np.save(f, Indv_cluster_total_UMIs)
        
    # Calculate Expression Matching quntiles
    Gene_Total_Exp = Gene_Total_Exp/Total_N_Cells
    Gene_Total_Exp.name="PerCellExp"
    Gene_Total_Exp_DF = pd.DataFrame(data=Gene_Total_Exp)
    Gene_Total_Exp_DF = Gene_Total_Exp_DF.sort_values("PerCellExp")
    Gene_Total_Exp_DF["Rank"] = [1+x for x in range(Gene_Total_Exp_DF.shape[0])] # compute Rank
    Gene_Total_Exp_DF["quantile"] = Gene_Total_Exp_DF["Rank"]/Gene_Total_Exp_DF.shape[0]
    Gene_Total_Exp_DF.to_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/ABC_LogUMI.Match.10xV3.csv")

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help = '')
    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    ClusterAnn = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label")

    pool = multiprocessing.Pool(processes=5)
    #clusters = ClusterAnn.index.values[110:130]
    clusters = ClusterAnn.index.values
    print(clusters)
    DatDIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/Cluster_GeneXCell_UMI/"
    results = pool.starmap(processCluster, [(cluster, DatDIR) for cluster in clusters])
    pool.close()
    pool.join()
    CollectRes(clusters, results)
    return

def main2():
    args = GetOptions()
    ClusterAnn = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label")

    pool = multiprocessing.Pool(processes=10)
    #clusters = ClusterAnn.index.values[110:130]
    clusters = ClusterAnn.index.values[5200:]
    print(clusters)
    DatDIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/Cluster_GeneXCell_UMI/"
    results = pool.starmap(processClusterV2, [(cluster, DatDIR) for cluster in clusters])
    pool.close()
    pool.join()
    return


def main3():
    args = GetOptions()
    ClusterAnn = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label")

    DatDIR = "/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/Cluster_GeneXCell_UMI/"
    clusters = ClusterAnn.index.values[5200:]
    print(clusters)
    for cluster in clusters:
        processCluster(cluster, DatDIR)
    return


if __name__=='__main__':
    #main()
    main2()
