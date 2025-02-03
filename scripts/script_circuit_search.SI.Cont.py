import argparse
import sys
sys.path.insert(1, '../src')
from ASD_Circuits import *


def run_CircuitOpt2(g, EdgeWeightsDict, BiasDF, weighted, directed, topN):
    CandidateNodes = BiasDF.head(topN).index.values
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if np.random.rand() > 0.5:
            Init_States[idx] = 0
    ins = MostCohesiveCirtuis(Init_States, g, CandidateNodes,
                              EdgeWeightsDict, Weighted=weighted, Direction=directed)
    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1
    ins.Tmin = 0.00001
    Tmps, Energys, state, e = ins.anneal()
    print("Done")
    return state

def FindInitState_old(tmpDF, size, BiasLim):
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

def FindInitState(tmpDF, size, BiasLim):
    STRs = tmpDF.index.values
    Biases = tmpDF["EFFECT"].values
    minBias = np.min(Biases)
    PsudoBiases = Biases - minBias + 1
    Probs = PsudoBiases/np.sum(PsudoBiases)
    Probs[-1] = 1 - np.sum(Probs[0:-1])
    init_STRs = np.random.choice(STRs, size=size, replace=False, p=Probs)
    return init_STRs

def RandInitState(tmpDF, size, BiasLim):
    STRs = tmpDF.index.values
    init_STRs = np.random.choice(STRs, size=size, replace=False)
    return init_STRs

def run_CircuitOpt(adj_mat, InfoMat, BiasDF, topN=100, keepN=30, minbias=-1):
    tmpBiasDF = BiasDF.head(topN)
    CandidateNodes = tmpBiasDF.index.values
    Init_States = np.zeros(topN)
    #init_STRs, rand_count = FindInitState(tmpBiasDF, keepN, minbias)
    init_STRs = FindInitState(tmpBiasDF, keepN, minbias)
    for i, _STR in enumerate(CandidateNodes):
        if _STR in init_STRs:
            Init_States[i] = 1
    ins = CircuitSearch_SA_InfoContent_v2(tmpBiasDF, Init_States, adj_mat, InfoMat, CandidateNodes, minbias=minbias)
    #ins = CircuitSearch_SA_Connectivity(BiasDF, Init_States, adj_mat, InfoMat, CandidateNodes, minbias=minbias)
    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1e-1
    ins.Tmin = 1e-6
    ins.steps = 50000
    Tmps, Energys, state, e = ins.anneal()
    res = CandidateNodes[np.where(state == 1)[0]]
    score = -e
    return res, score


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--biaslim', type=str, help="bias lim file")
    parser.add_argument('-b', '--bias', type=str, help='Bias File')
    parser.add_argument('-q', '--qn_bias', type=str,
                        help='Bias File quntail from (ASD Bias File)')
    parser.add_argument('-s', '--suffix', type=str, help='suffix of output')
    parser.add_argument('-g', '--graph', type=str,
                        help='adj mat of graph (permuteted)')
    parser.add_argument('--mat', type=str, help="info matrix")
    parser.add_argument('-t', '--topN', type=int,
                        help="Size of top N bias Node in circuit")
    parser.add_argument('-d', '--dir', type=str,
                        help="Directory to save the results")
    parser.add_argument('--runtimes', type=int, default=10,
                        help="Repeat Times for SA")
    args = parser.parse_args()
    return args


def BiasTransferSA(args):
    topN = args.topN
    runT = args.runtimes
    # Load Connectome
    adj_mat = pd.read_csv(args.graph, index_col=0)
    InfoMat = pd.read_csv(args.mat, index_col=0)

    # Load ExpBias
    BiasDF = pd.read_csv(args.bias, index_col="STR")
    BiasDF2 = pd.read_csv(args.qn_bias, index_col="STR")
    BiasDF["EFFECT"] = BiasDF2["EFFECT"].values
    subDIR = args.bias.split("/")[-1].rstrip(".csv")
    print("------>", subDIR)
    DIR = args.dir + "/" + subDIR + "/"
    if not os.path.isdir(DIR):
        os.makedirs(DIR)
    suffix = args.suffix

    
    topN = args.topN
    #keepN = args.keepN
    #minbias = args.minbias
    #print("InpFil: ", args.bias)
    reader = csv.reader(open(args.biaslim))
    for row in reader:
        keepN, minbias = row
        keepN = int(keepN)
        minbias = float(minbias)
        print("topN: {}\tkeepN: {}\t minbias: {}".format(topN, keepN, minbias))
        fout = open("{}/SA.{}.topN_{}-keepN_{}-minbias_{}.txt".format(DIR,
                    subDIR, topN, keepN, minbias), 'wt')
        for i in range(runT):
            res, score = run_CircuitOpt(
                adj_mat, InfoMat, BiasDF, topN=topN, keepN=keepN, minbias=minbias)
            meanbias = BiasDF.loc[res, "EFFECT"].mean()
            fout.write("{}\t{}\t".format(score, meanbias) + ",".join(res) + "\n")
        fout.close()

    print()
    print("Done")

def main():
    args = GetOptions()
    BiasTransferSA(args)


main()
