from ASD_Circuits import *

def run_CircuitOpt(g, EdgeWeightsDict, BiasDF):
    CandidateNodes = BiasDF.head(50).index.values
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if np.random.rand() > 0.5:
            Init_States[idx] = 0
    ins = MostCohesiveCirtuis(Init_States, g, CandidateNodes, EdgeWeightsDict, Direction=True, Weighted=True)
    ins.copy_strategy = "deepcopy"
    ins.Tmax=1
    ins.Tmin=0.00001
    Tmps, Energys, state, e = ins.anneal()
    print("Done")
    return state

def main():
    # Load Connectome
    g = LoadConnectome2(ConnFil="../dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv")
    EdgeWeightsDict = EdgeDict(g, keyon="label")

    # Load ExpBias
    BiasDF = pd.read_csv("dat/Jon_data/exp_bias-match-specific.csv")
    BiasDF.columns = ["STR", "EFFECT", "Rank", "NGene"]
    BiasDF = BiasDF.set_index("STR")
    
    csv_out = csv.writer(open("dat/circuits/ASD_Pad1_SA.WD.csv", 'wt'))
    csv_out.writerow(BiasDF.head(50).index.values)
    for i in range(100):
        state = run_CircuitOpt(g, EdgeWeightsDict, BiasDF)
        csv_out.writerow(state)

main()
