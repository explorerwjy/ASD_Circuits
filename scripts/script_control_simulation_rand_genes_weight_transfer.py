#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_control_simulation.py
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *


class script_control_simulation:
    def __init__(self, args):
        args.Noneg = False
        self.Mat = args.matrix
        self.idx = str(args.input)
        self.weights = args.weights
        self.location = args.location
        self.Noneg = args.Noneg
        self.prob = args.prob

    def run(self):
        if not os.path.exists(self.location):
            os.makedirs(self.location)
        ExpMatNorm = pd.read_csv(self.Mat, index_col=0)
        WeightDF = pd.read_csv(self.weights, header=None)
        Gene_Weights = [1] * len(WeightDF[1].values)
        #Gene_Weights = WeightDF[1].values
        for i in range(10):
            if self.prob == None:
                Genes = np.random.choice(
                    ExpMatNorm.index.values, size=len(Gene_Weights))
            else:
                Gene2Prob = pd.read_csv(self.prob, index_col=0)
                Genes = np.random.choice(Gene2Prob.index.values, size=len(
                    Gene_Weights), p=Gene2Prob["Prob"].values)
            tmp_dict = dict(zip(Genes, Gene_Weights))
            ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMatNorm, tmp_dict, Method=1, NonNeg=False, csv_fil="{}/cont.bias.{}.{}.csv".format(self.location, self.idx, i))
            #List2Fil(Genes, "{}/cont.genes.{}.{}csv".format(self.location, self.idx, i))

    def run_subsampleSibling(self):
        SibWeightDF = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/sibling_weights_LGD_Dmis.csv", header=None)
        SibGenes = SibWeightDF[0].values
        if not os.path.exists(self.location):
            os.makedirs(self.location)
        ExpMatNorm = pd.read_csv(self.Mat, index_col=0)
        WeightDF = pd.read_csv(self.weights, header=None)
        #Gene_Weights = [1] * len(WeightDF[1].values)
        Gene_Weights = WeightDF[1].values

        # Adjust Prob
        if self.prob != None:
            Gene2Prob = pd.read_csv(self.prob, index_col=0)
            SibGenes = [g for g in Gene2Prob.index.values if g in SibGenes]
            Gene2Prob = Gene2Prob.loc[SibGenes, :]
            probs = Gene2Prob["Prob"].values
            total = np.sum(probs)
            probs = probs/total
            probs[-1] = 1 - np.sum(probs[:-1])
            Gene2Prob["Prob"] = probs

        for i in range(100):
            if self.prob == None:
                Genes = np.random.choice(
                    ExpMatNorm.index.values, size=len(Gene_Weights))
            else:
                Genes = np.random.choice(Gene2Prob.index.values, size=len(
                    Gene_Weights), p=Gene2Prob["Prob"].values)
            tmp_dict = dict(zip(Genes, Gene_Weights))
            ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMatNorm, tmp_dict, Method=1, NonNeg=False, csv_fil="{}/cont.bias.{}.{}.csv".format(self.location, self.idx, i))
            List2Fil(Genes, "{}/cont.genes.{}.{}csv".format(self.location, self.idx, i))

    def run_subsampleSiblingAbvBias(self, top50_biaslim):
        SibWeightDF = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/sibling_weights_LGD_Dmis.csv", header=None)
        SibGenes = SibWeightDF[0].values
        if not os.path.exists(self.location):
            os.makedirs(self.location)
        ExpMatNorm = pd.read_csv(self.Mat, index_col=0)
        WeightDF = pd.read_csv(self.weights, header=None)
        #Gene_Weights = [1] * len(WeightDF[1].values)
        Gene_Weights = WeightDF[1].values

        # Adjust Prob
        if self.prob != None:
            Gene2Prob = pd.read_csv(self.prob, index_col=0)
            SibGenes = [g for g in Gene2Prob.index.values if g in SibGenes]
            Gene2Prob = Gene2Prob.loc[SibGenes, :]
            probs = Gene2Prob["Prob"].values
            total = np.sum(probs)
            probs = probs/total
            probs[-1] = 1 - np.sum(probs[:-1])
            Gene2Prob["Prob"] = probs
    
        Accept_count = 0
        for i in range(10000):
            if self.prob == None:
                Genes = np.random.choice(
                    ExpMatNorm.index.values, size=len(Gene_Weights))
            else:
                Genes = np.random.choice(Gene2Prob.index.values, size=len(
                    Gene_Weights), p=Gene2Prob["Prob"].values)
            tmp_dict = dict(zip(Genes, Gene_Weights))
            #ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMatNorm, tmp_dict, Method=1, NonNeg=False, csv_fil="{}/cont.bias.{}.{}.csv".format(self.location, self.idx, i))
            ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMatNorm, tmp_dict, Method=1, NonNeg=False)
            print(ASD_Meta_Spec.head(46)["EFFECT"].mean())
            if ASD_Meta_Spec.head(46)["EFFECT"].mean() > top50_biaslim:
                Accept_count += 1
                ASD_Meta_Spec.to_csv("{}/cont.bias.{}.{}.csv".format(self.location, self.idx, i), index=False)
            #List2Fil(Genes, "{}/cont.genes.{}.{}csv".format(self.location, self.idx, i))


    def run_subsampleSiblingCircuitScoreLim(self, mat, ciruitscore_lim=0.5746725449959333, topN=46):
        #adj_mat = pd.read_csv(args.graph, index_col=0)
        adj_mat = pd.read_csv(ConnFil, index_col=0)
        print("#######################################")
        print(mat)
        print("#######################################")
        InfoMat = pd.read_csv(mat, index_col=0)
        ProbMat1 = np.exp2(-InfoMat)
        ProbMat1[ProbMat1==1] = 0
        ProbMat2 = 1-ProbMat1

        SibWeightDF = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/sibling_weights_LGD_Dmis.csv", header=None)
        SibGenes = SibWeightDF[0].values
        if not os.path.exists(self.location):
            os.makedirs(self.location)
        ExpMatNorm = pd.read_csv(self.Mat, index_col=0)
        WeightDF = pd.read_csv(self.weights, header=None)
        #Gene_Weights = [1] * len(WeightDF[1].values)
        Gene_Weights = WeightDF[1].values

        # Adjust Prob
        if self.prob != None:
            Gene2Prob = pd.read_csv(self.prob, index_col=0)
            SibGenes = [g for g in Gene2Prob.index.values if g in SibGenes]
            Gene2Prob = Gene2Prob.loc[SibGenes, :]
            probs = Gene2Prob["Prob"].values
            total = np.sum(probs)
            probs = probs/total
            probs[-1] = 1 - np.sum(probs[:-1])
            Gene2Prob["Prob"] = probs
    
        Accept_count = 0
        for i in range(10000):
        #for i in range(100):
            if self.prob == None:
                Genes = np.random.choice(
                    ExpMatNorm.index.values, size=len(Gene_Weights))
            else:
                Genes = np.random.choice(Gene2Prob.index.values, size=len(
                    Gene_Weights), p=Gene2Prob["Prob"].values)
            tmp_dict = dict(zip(Genes, Gene_Weights))
            #ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMatNorm, tmp_dict, Method=1, NonNeg=False, csv_fil="{}/cont.bias.{}.{}.csv".format(self.location, self.idx, i))
            ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMatNorm, tmp_dict, Method=1, NonNeg=False)
            #print(ASD_Meta_Spec.head(46)["EFFECT"].mean())
            top_strs = ASD_Meta_Spec.head(46)["STR"]
            cir_score = ScoreCircuit_v7(top_strs, adj_mat, ProbMat1, ProbMat2)
            if cir_score > ciruitscore_lim:
                Accept_count += 1
                ASD_Meta_Spec.to_csv("{}/cont.bias.{}.{}.csv".format(self.location, self.idx, i), index=False)

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=str, default="dat/z2.all.genes.str.mat.csv", help='Which Col from Match Set')
    parser.add_argument('--Noneg', type=bool, default=False,
                        help='Which Col from Match Set')
    parser.add_argument('--weights', type=str, help='Which Col from Match Set')
    parser.add_argument('--prob', type=str, help='Fil with Probability of each gene being sampled [optional]')
    parser.add_argument('-i', '--input', type=int,
                        required=True,  help='Index of simulation')
    parser.add_argument('-l', '--location', type=str,
                        default="dat/SimulateControlBiasDefault/",  help='location to store results')

    parser.add_argument('--mat', type=str)
    parser.add_argument('--graph', type=str)
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = script_control_simulation(args)
    #ins.run()
    ins.run_subsampleSibling()
    #ins.run_subsampleSiblingAbvBias(0.35)
    #ins.run_subsampleSiblingCircuitScoreLim(args.mat)

    return


if __name__ == '__main__':
    main()
    print("Done")
