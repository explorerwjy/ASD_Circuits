import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *
import argparse

def FindInitState(tmpDF, size, BiasLim):
    STRs = tmpDF.index.values
    Biases = tmpDF["EFFECT"].values
    minBias = np.min(Biases)
    PsudoBiases = Biases - minBias + 1
    Probs = PsudoBiases/np.sum(PsudoBiases)
    Probs[-1] = 1 - np.sum(Probs[0:-1])
    init_STRs = np.random.choice(STRs, size=size, replace=False, p=Probs)
    return init_STRs

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int)
    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    index = args.index
    
    BiasDF = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR")
    InfoMat = pd.read_csv("../dat/allen-mouse-conn/ScoreingMat_jw_v3/InfoMat.Ipsi.csv", index_col=0)
    adj_mat = pd.read_csv(ConnFil, index_col=0)

    minbias = 0.37
    topN = 213
    keepN = 46

    tmpDF = BiasDF.head(topN)
    STRs = tmpDF.index.values
    # Determine init state
    init_STRs = FindInitState(tmpDF, keepN, minbias)

    STRs = BiasDF.index.values
    topN = 213
    init_state = np.zeros(topN)
    for i, _STR in enumerate(STRs):
        if _STR in init_STRs:
            init_state[i] = 1
    ins = CircuitSearch_SA_InfoContent_v2(BiasDF, init_state, adj_mat, 
                                   InfoMat, STRs, minbias=minbias)

    ins.copy_strategy = "deepcopy"
    ins.Tmax=1e-1
    ins.Tmin=1e-7
    ins.steps=50000
    Tmps, Energys, state, e = ins.anneal()
    res = STRs[np.where(state==1)[0]]
    score = -e
    OutDir = "/home/jw3514/Work/ASD_Circuits/dat/Circuits/SA/test_asd_annealingParam/"
    fname = OutDir + "MinBias{}-Tmin{}-Tmax{}-NSteps{}.{}.txt".format(minbias, ins.Tmin, ins.Tmax, ins.steps, index)
    with open(fname, 'wt') as fout:
        fout.write(str(tmpDF.loc[res, "EFFECT"].mean()) + "\n")
        fout.write(str(ScoreCircuit_SI_Joint(res, InfoMat)) + "\n")
        fout.write(str(-e) + "\n")
        fout.write(",".join(res)+"\n")

main()



