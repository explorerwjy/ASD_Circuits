from ASD_Circuits import *
import argparse


def run_CircuitOpt(g, EdgeWeightsDict, BiasDF, weighted, directed, topN):
    CandidateNodes = BiasDF.head(topN).index.values
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if np.random.rand() > 0.5:
            Init_States[idx] = 0
    ins = MostCohesiveCirtuis(Init_States, g, CandidateNodes,
                              EdgeWeightsDict, Weighted=weighted, Direction=directed)
    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    Tmps, Energys, state, e = ins.anneal()
    print("Done")
    return state


def run_CircuitOpt_keepN(g, EdgeWeightsDict, BiasDF, weighted, directed, topN, keepN):
    CandidateNodes = BiasDF.head(topN).index.values
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if idx > keepN - 1:
            Init_States[idx] = 0
    ins = MostCohesiveCirtuisGivenNCandidates(
        Init_States, g, CandidateNodes, EdgeWeightsDict, Weighted=weighted, Direction=directed)
    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    Tmps, Energys, state, e = ins.anneal()
    print("Done")
    return state


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Bias File')
    parser.add_argument('-w', '--weighted', type=int,
                        help='use weighted cohesivebess')
    parser.add_argument('-d', '--directed', type=int,
                        help='use directed cohesivebess')
    parser.add_argument('-n', '--topN', type=int,
                        default=50, help='top N as candidate STRs')
    parser.add_argument('-o', '--out', type=str,
                        help='output file name')
    parser.add_argument('-g', '--graph', type=str, default="../dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv",
                        help='adj mat of graph (permuteted)')
    parser.add_argument('-k', '--keepN', type=int,
                        help="Size of node kept in circuit")
    args = parser.parse_args()
    return args


def main():
    args = GetOptions()
    if args.weighted == 1:
        weighted = True
    else:
        weighted = False
    if args.directed == 1:
        directed = True
    else:
        directed = False
    topN = args.topN

    # Load Connectome
    adj_mat = pd.read_csv(args.graph, index_col=0)
    g = LoadConnectome2(ConnFil=adj_mat)
    EdgeWeightsDict = EdgeDict(g, keyon="label")

    # Load ExpBias
    BiasDF = pd.read_csv(args.input, index_col="STR")
    #BiasDF.columns = ["STR", "EFFECT", "Rank", "NGene"]
    #BiasDF = BiasDF.set_index("STR")

    # Make suffix
    #suffix = args.suffix
    #if weighted or directed:
    #    suffix = suffix + "."
    #if weighted:
    #    suffix += "w"
    #if directed:
    #    suffix += "d"
    #suffix += ".{}".format(topN)
    foutname = args.out
    if ".csv" not in foutname:
        foutname += ".csv"
    csv_out = csv.writer(open(foutname, 'wt'))
    csv_out.writerow(BiasDF.head(topN).index.values)
    # for keepN in range(49, 2, -1):
    #	state = run_CircuitOpt(g, EdgeWeightsDict, BiasDF, weighted, directed, topN, keepN)
    #	csv_out.writerow(state)
    for i in range(100):
        state = run_CircuitOpt(g, EdgeWeightsDict, BiasDF, weighted, directed, topN)
        csv_out.writerow(state)


main()
