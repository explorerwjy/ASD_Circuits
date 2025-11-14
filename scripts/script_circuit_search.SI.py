import argparse
import sys
sys.path.insert(1, '../src')
from ASD_Circuits import *

#res = run_CircuitOpt(adj_mat, BiasDF, topN=topN, keepN=keepN, minbias=minbias)

def FindInitState(tmpDF, size, BiasLim):
    STRs = tmpDF.index.values
    Biases = tmpDF["EFFECT"].values
    minBias = np.min(Biases)
    PsudoBiases = Biases - minBias + 1
    PsudoBiases = np.power(PsudoBiases, BiasLim*150-17)
    Probs = PsudoBiases/np.sum(PsudoBiases)
    Probs[-1] = 1 - np.sum(Probs[0:-1])
    rand_count = 0
    while 1:
        init_STRs = np.random.choice(STRs, size=size, replace=False, p=Probs)
        avg_bias_init_STRs = tmpDF.loc[init_STRs, "EFFECT"].mean()
        if avg_bias_init_STRs >= BiasLim:
            return init_STRs, rand_count
        else:
            rand_count += 1

def run_CircuitOpt(BiasDF, adj_mat, InfoMat, topN=213, keepN=46, minbias=-1, measure="SI"):
    tmpBiasDF = BiasDF.head(topN)
    CandidateNodes = tmpBiasDF.index.values
    Init_States = np.zeros(topN)
    init_STRs, rand_count = FindInitState(tmpBiasDF, keepN, minbias)
    for i, _STR in enumerate(CandidateNodes):
        if _STR in init_STRs:
            Init_States[i] = 1
    print(Init_States)
    print(CandidateNodes)
    if measure == "SI":
        ins = CircuitSearch_SA_InfoContent(tmpBiasDF, Init_States, adj_mat,
                                           InfoMat, CandidateNodes, minbias=minbias)
    elif measure == "Connectivity":
        ins = CircuitSearch_SA_Connectivity(BiasDF, Init_States, adj_mat, InfoMat, CandidateNodes, minbias=minbias)

    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    ins.updates = 0  # Suppress verbose output
    Tmps, Energys, state, e = ins.anneal()
    res = CandidateNodes[np.where(state == 1)[0]]
    #score = Energys[-1]
    score = - e
    return res, score


def run_CircuitOpt_old(BiasDF, adj_mat, InfoMat, topN=100, keepN=30, minbias=-1, measure="SI"):
    CandidateNodes = BiasDF.head(topN).index.values
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if idx > keepN-1:
            Init_States[idx] = 0
    if measure == "SI":
        ins = CircuitSearch_SA_InfoContent(BiasDF, Init_States, adj_mat,
                                           InfoMat, CandidateNodes, minbias=minbias)
    elif measure == "Connectivity":
        ins = CircuitSearch_SA_Connectivity(BiasDF, Init_States, adj_mat, InfoMat, CandidateNodes, minbias=minbias)

    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    ins.updates = 0  # Suppress verbose output
    Tmps, Energys, state, e = ins.anneal()
    res = CandidateNodes[np.where(state == 1)[0]]
    #score = Energys[-1]
    score = - e
    return res, score


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bias', type=str, help='Bias File')
    parser.add_argument('-s', '--suffix', type=str, help='suffix of output')
    parser.add_argument('-g', '--graph', type=str,
                        help='adj mat of graph (permuteted)')
    parser.add_argument('--mat', type=str, help="info matrix")
    parser.add_argument('-t', '--topN', type=int, default = 213,
                        help="Size of top N bias Node in circuit")
    parser.add_argument('-k', '--keepN', type=int,
                        help="Size of node kept in circuit")
    parser.add_argument('-d', '--dir', type=str,
                        help="Directory to save the results")
    parser.add_argument('--minbias', type=float,
                        help="Minimun Bias for average structure circuits")
    parser.add_argument('--runtimes', type=int, default=5,
                        help="Repeat Times for SA")
    parser.add_argument('--measure', default="SI", type=str,
                        help="measure of circuit score [SI/Connectivity]")
    args = parser.parse_args()
    return args


def main():
    args = GetOptions()
    topN = args.topN
    runT = args.runtimes
    suffix = args.suffix
    topN = args.topN
    keepN = args.keepN
    minbias = args.minbias
    # Load Connectome
    print("XXXXX-----|{}".format(args.mat))
    adj_mat = pd.read_csv(args.graph, index_col=0)
    InfoMat = pd.read_csv(args.mat, index_col=0)

    # Load ExpBias
    BiasDF = pd.read_csv(args.bias, index_col="STR")
    DIR = args.dir
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    print("InpFil: ", args.bias)
    print("topN: {}\tkeepN: {}\t minbias: {}".format(topN, keepN, minbias))
    fout = open("{}/SA.{}.topN_{}-keepN_{}-minbias_{}.txt".format(DIR,
                suffix, topN, keepN, minbias), 'wt')
    for i in range(runT):
        res, score = run_CircuitOpt(BiasDF, adj_mat, InfoMat, topN=topN, keepN=keepN, minbias=minbias, measure=args.measure)
        meanbias = BiasDF.loc[res, "EFFECT"].mean()
        fout.write("{}\t{}\t".format(score, meanbias) + ",".join(res) + "\n")
    fout.close()
    print()
    print("Done")

main()
