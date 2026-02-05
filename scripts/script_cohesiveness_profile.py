#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# script_cohesiveness_profile.py
# June27 2022: Use Infomation Score
#========================================================================================================

import argparse
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ASD_Circuits import *

class script_cohesiveness_profile:
    def __init__(self, args):
        self.biasDF = args.biasDF
        self.contDir = args.contDir
        self.OutName = args.OutName
        self.scoreMat = pd.read_csv(args.scoreMat, index_col=0)
        #self.scoreMat2 = pd.read_csv(args.scoreMat2, index_col=0)
        self.scoreMethod = args.method

    def run(self):
        InfoMat = self.scoreMat
        BiasDF = pd.read_csv(self.biasDF, index_col="STR")
        str2reg, reg2str = LoadSTR2REG()
        topNs = list(range(200, 5, -1))
        STR_Ranks = pd.read_csv(self.biasDF, index_col="STR").index.values
        asd_topN_score = []
        for topN in topNs:
            top_strs = STR_Ranks[:topN]
            asd_score = ScoreCircuit_SI_Joint(top_strs, InfoMat)
            asd_topN_score.append(asd_score)
        asd_topN_score = np.array(asd_topN_score)

        ASD_Cont_Scores = []
        Ncont = 10000
        for i, file in enumerate(os.listdir(self.contDir)):
            if file.startswith("cont.genes"):
                continue
            df = pd.read_csv(self.contDir + file, index_col="STR")
            topN_scores = []
            for topN in topNs:
                top_strs = df.index.values[:topN]
                cont_score = ScoreCircuit_SI_Joint(top_strs, InfoMat)
                topN_scores.append(cont_score)
            topN_scores = np.array(topN_scores)
            ASD_Cont_Scores.append(topN_scores)
            if i >= Ncont-1:
                break
        ASD_Cont_Scores = np.array(ASD_Cont_Scores)
        np.save("RankScore.{}.ASD.npy".format(self.OutName), asd_topN_score)
        np.save("RankScore.{}.Cont.npy".format(self.OutName), ASD_Cont_Scores)

    def run3(self):
        BiasDF = pd.read_csv(self.biasDF, index_col="STR")
        adj_mat = pd.read_csv(ConnFil, index_col=0)
        ProbMat1 = np.exp2(-self.scoreMat)
        ProbMat1[ProbMat1==1] = 0
        ProbMat2 = 1-ProbMat1 


        str2reg, reg2str = LoadSTR2REG()
        topNs = list(range(200, 5, -1))
        STR_Ranks = pd.read_csv(self.biasDF, index_col="STR").index.values
        asd_topN_cohe_1 = []
        for topN in topNs:
            top_strs = STR_Ranks[:topN]
            asd_score1 = ScoreCircuit_v7(top_strs, adj_mat, ProbMat1, ProbMat2)
            asd_topN_cohe_1.append(asd_score1)
        asd_topN_cohe_1 = np.array(asd_topN_cohe_1)

        ASD_Cont_Cohesivness_1 = []
        Ncont = 10000
        for i, file in enumerate(os.listdir(self.contDir)):
            if file.startswith("cont.genes"):
                continue
            df = pd.read_csv(self.contDir + file, index_col="STR")
            topN_cohe_1 = []
            topN_cohe_2 = []
            for topN in topNs:
                top_strs = df.index.values[:topN]
                score1 = ScoreCircuit_v7(top_strs, adj_mat, ProbMat1, ProbMat2)
                topN_cohe_1.append(score1)
            topN_cohe_1 = np.array(topN_cohe_1)
            ASD_Cont_Cohesivness_1.append(topN_cohe_1)
            if i >= Ncont-1:
                break
        ASD_Cont_Cohesivness_1 = np.array(ASD_Cont_Cohesivness_1)
        np.save("RankScore.{}.ASD.S1.npy".format(self.OutName), asd_topN_cohe_1)
        np.save("RankScore.{}.Cont.S1.npy".format(self.OutName), ASD_Cont_Cohesivness_1)

    def run2(self):
        BiasDF = pd.read_csv(self.biasDF, index_col="STR")
        #g = LoadConnectome2()
        adj_mat = pd.read_csv(ConnFil, index_col=0)
        ProbMat1 = np.exp2(-self.scoreMat)
        ProbMat1[ProbMat1==1] = 0
        ProbMat2 = 1-ProbMat1 


        str2reg, reg2str = LoadSTR2REG()
        topNs = list(range(200, 5, -1))
        #EWE, EWP = CohesivenessProfile(BiasDF, topNs=topNs, g=g)
        #PlotCohesivenessProfile(topNs, EWE, savefig="figs/cohesiveness-rank.{}".format(self.outname))
        STR_Ranks = pd.read_csv(self.biasDF, index_col="STR").index.values
        asd_topN_cohe_1 = []
        #asd_topN_cohe_2 = []
        for topN in topNs:
            top_strs = STR_Ranks[:topN]
            #asd_score1 = ScoreCircuit_v7(top_strs, adj_mat, ProbMat1, ProbMat2)
            asd_score1 = ScoreCircuit_v7(top_strs, adj_mat, ProbMat1, ProbMat2)
            asd_topN_cohe_1.append(asd_score1)
            #asd_topN_cohe_2.append(asd_score2)
        asd_topN_cohe_1 = np.array(asd_topN_cohe_1)
        #asd_topN_cohe_2 = np.array(asd_topN_cohe_2)

        #cont_dir = "../dat/Unionize_bias/ASD_Sim/"
        ASD_Cont_Cohesivness_1 = []
        #ASD_Cont_Cohesivness_2 = []
        Ncont = 10000
        for i, file in enumerate(os.listdir(self.contDir)):
            if file.startswith("cont.genes"):
                continue
            df = pd.read_csv(self.contDir + file, index_col="STR")
            topN_cohe_1 = []
            topN_cohe_2 = []
            for topN in topNs:
                top_strs = df.index.values[:topN]
                score1 = ScoreCircuit_v7(top_strs, adj_mat, ProbMat1, ProbMat2)
                topN_cohe_1.append(score1)
                #topN_cohe_2.append(score2)
            topN_cohe_1 = np.array(topN_cohe_1)
            #topN_cohe_2 = np.array(topN_cohe_2)
            ASD_Cont_Cohesivness_1.append(topN_cohe_1)
            #ASD_Cont_Cohesivness_2.append(topN_cohe_2)
            if i >= Ncont-1:
                break
        ASD_Cont_Cohesivness_1 = np.array(ASD_Cont_Cohesivness_1)
        #ASD_Cont_Cohesivness_2 = np.array(ASD_Cont_Cohesivness_2)
        #np.save("RankScore.{}.ASD.In.npy".format(self.OutName), asd_topN_cohe_1)
        #np.save("RankScore.{}.ASD.InOut.npy".format(self.OutName), asd_topN_cohe_2)
        #np.save("RankScore.{}.Cont.In.npy".format(self.OutName), ASD_Cont_Cohesivness_1)
        #np.save("RankScore.{}.Cont.InOut.npy".format(self.OutName), ASD_Cont_Cohesivness_2)
        np.save("RankScore.{}.ASD.S1.npy".format(self.OutName), asd_topN_cohe_1)
        #np.save("RankScore.{}.ASD.S2.npy".format(self.OutName), asd_topN_cohe_2)
        np.save("RankScore.{}.Cont.S1.npy".format(self.OutName), ASD_Cont_Cohesivness_1)
        #np.save("RankScore.{}.Cont.S2.npy".format(self.OutName), ASD_Cont_Cohesivness_2)
                
    def run_ipsi_contra(self):
        BiasDF = pd.read_csv(self.biasDF, index_col="STR")
        adj_mat = pd.read_csv(ConnFil, index_col=0)
        ProbMat1 = np.exp2(-self.scoreMat)
        ProbMat1[ProbMat1==1] = 0
        ProbMat2 = 1-ProbMat1 

        str2reg, reg2str = LoadSTR2REG()
        topNs = list(range(200, 5, -1))
        STR_Ranks = pd.read_csv(self.biasDF, index_col="STR").index.values
        asd_topN_cohe = []
        for topN in topNs:
            top_strs = STR_Ranks[:topN]
            asd_score = ScoreCircuit_SI_ipsi_contra(top_strs, adj_mat, ProbMat1, ProbMat2)
            asd_topN_cohe.append(asd_score)
        asd_topN_cohe = np.array(asd_topN_cohe)

        ASD_Cont_Cohesivness = []
        Ncont = 10000
        for i, file in enumerate(os.listdir(self.contDir)):
            if file.startswith("cont.genes"):
                continue
            df = pd.read_csv(self.contDir + file, index_col="STR")
            topN_cohe = []
            for topN in topNs:
                top_strs = df.index.values[:topN]
                score = ScoreCircuit_SI_ipsi_contra(top_strs, adj_mat, ProbMat1, ProbMat2)
                topN_cohe.append(score)
            topN_cohe = np.array(topN_cohe)
            ASD_Cont_Cohesivness.append(topN_cohe)
            if i >= Ncont-1:
                break
        ASD_Cont_Cohesivness = np.array(ASD_Cont_Cohesivness)
        np.save("RankScores/RankScore.{}.ASD.ipsi_contra.npy".format(self.OutName), asd_topN_cohe)
        np.save("RankScores/RankScore.{}.Cont.ipis_contra.npy".format(self.OutName), ASD_Cont_Cohesivness)
                
def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--biasDF', type=str, help = 'Bias Fil of ASD')
    parser.add_argument('-c', '--contDir', type=str, help= 'control Directory')
    parser.add_argument('-m', '--scoreMat', type=str, help= 'score Matrix to use')
    parser.add_argument('-m2', '--scoreMat2', type=str, help= 'score Matrix to use')
    parser.add_argument('--method', type=int, help = 'Method to calculate score: [1: inside] [2: inside / outside]')
    parser.add_argument('-o','--OutName', type=str, help = 'Name of Output fil')
    args = parser.parse_args()
    
    return args

def main():
    args = GetOptions()
    ins = script_cohesiveness_profile(args)
    ins.run()
    #ins.run_ipsi_contra()
    return

if __name__=='__main__':
    main()
