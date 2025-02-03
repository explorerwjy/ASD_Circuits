import sys
sys.path.insert(1, '../src')
from ASD_Circuits import *
import argparse

def run_CircuitOpt2(g, EdgeWeightsDict, BiasDF, weighted, directed, topN):
	CandidateNodes = BiasDF.head(topN).index.values
	Init_States = np.ones(len(CandidateNodes))
	for idx in range(len(Init_States)):
		if np.random.rand() > 0.5:
			Init_States[idx] = 0
	ins = MostCohesiveCirtuis(Init_States, g, CandidateNodes, EdgeWeightsDict, Weighted=weighted, Direction=directed)
	ins.copy_strategy = "deepcopy"
	ins.Tmax=1
	ins.Tmin=0.00001
	Tmps, Energys, state, e = ins.anneal()
	print("Done")
	return state

def run_CircuitOpt(adj_mat, ProbMat1, ProbMat2, BiasDF, topN=100, keepN=30, minbias=-1):
	CandidateNodes = BiasDF.head(topN).index.values
	Init_States = np.ones(len(CandidateNodes))
	for idx in range(len(Init_States)):
		if idx > keepN-1:
			Init_States[idx] = 0
	#ins = MostCohesiveCirtuisGivenNCandidates(BiasDF, Init_States, g, CandidateNodes, EdgeWeightsDict, Weighted=weighted, Direction=directed, minbias=minbias)
	#ins = MostCohesiveCirtuisGivenNCandidates_LikelihoodScore(BiasDF, Init_States, adj_mat, CandidateNodes, minbias=minbias)
	ins = CircuitSearch_SA_InfoContent(BiasDF, Init_States, adj_mat, ProbMat1, ProbMat2, CandidateNodes, minbias=minbias)
	ins.copy_strategy = "deepcopy"
	ins.Tmax=1e-2
	ins.Tmin=5e-5
	Tmps, Energys, state, e = ins.anneal()
	res = CandidateNodes[np.where(state==1)[0]]
	score = Energys[-1]
	return res, score

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--bias', type=str, help='Bias File')
	parser.add_argument('-q', '--qn_bias', type=str, help='Bias File quntail from (ASD Bias File)')
	parser.add_argument('-s', '--suffix', type=str, help='suffix of output')
	parser.add_argument('-g', '--graph', type=str, help='adj mat of graph (permuteted)')
	parser.add_argument('--mat', type=str, help="info matrix")
	parser.add_argument('-t', '--topN', type=int, help="Size of top N bias Node in circuit")
	parser.add_argument('-k', '--keepN', type=int, help="Size of node kept in circuit")
	parser.add_argument('-d', '--dir', type=str, help="Directory to save the results")
	parser.add_argument('--minbias', type=float, help="Minimun Bias for average structure circuits")
	parser.add_argument('--runtimes', type=int, default=10, help="Repeat Times for SA")
	args = parser.parse_args()
	return args

def main():
	args = GetOptions()
	topN = args.topN
	runT = args.runtimes
	# Load Connectome
	adj_mat = pd.read_csv(args.graph, index_col=0)
	InfoMat = pd.read_csv(args.mat, index_col=0)
	ProbMat1 = np.exp2(-InfoMat)
	ProbMat1[ProbMat1==1] = 0
	ProbMat2 = 1-ProbMat1

	# Load ExpBias
	BiasDF = pd.read_csv(args.bias, index_col="STR")
	BiasDF2 = pd.read_csv(args.qn_bias, index_col="STR")
	BiasDF["EFFECT"] = BiasDF2["EFFECT"].values
	DIR = args.dir
	if not os.path.isdir(DIR):
		os.makedirs(DIR)
	suffix=args.suffix
	topN = args.topN
	keepN = args.keepN
	minbias = args.minbias
	print("InpFil: ",args.bias)
	print("topN: {}\tkeepN: {}\t minbias: {}".format(topN, keepN, minbias))
	fout = open("{}/SA.{}.topN_{}-keepN_{}-minbias_{}.txt".format(DIR, suffix, topN, keepN, minbias), 'wt')
	for i in range(runT):
		res, score = run_CircuitOpt(adj_mat, ProbMat1, ProbMat2, BiasDF, topN=topN, keepN=keepN, minbias=minbias)
		meanbias = BiasDF.loc[res, "EFFECT"].mean()
		fout.write("{}\t{}\t".format(score, meanbias) + ",".join(res) + "\n")
	fout.close()
	print()
	print("Done")

main()
