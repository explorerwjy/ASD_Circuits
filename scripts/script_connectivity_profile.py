#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# script_cohesiveness_profile.py
# June27 2022: Use Infomation Score
# January30 2023: Add Connectivity
#========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
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
            #asd_score = ScoreCircuit_SI_Joint(top_strs, InfoMat)
            asd_score = ScoreCircuit_NEdges(top_strs, InfoMat)
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
                #cont_score = ScoreCircuit_SI_Joint(top_strs, InfoMat)
                cont_score = ScoreCircuit_NEdges(top_strs, InfoMat)
                topN_scores.append(cont_score)
            topN_scores = np.array(topN_scores)
            ASD_Cont_Scores.append(topN_scores)
            if i >= Ncont-1:
                break
        ASD_Cont_Scores = np.array(ASD_Cont_Scores)
        np.save("RankConn.{}.ASD.npy".format(self.OutName), asd_topN_score)
        np.save("RankConn.{}.Cont.npy".format(self.OutName), ASD_Cont_Scores)
                
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
    return

if __name__=='__main__':
    main()
