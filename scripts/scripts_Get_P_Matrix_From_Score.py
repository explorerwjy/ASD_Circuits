# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# scripts_Get_P_Matrix_From_Score.py
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *

class scripts_Get_P_Matrix_From_Score:
    def __init__(self, args):
        self.start = args.start
        self.step = args.step
        self.Cont_SI = np.load(args.input)
        self.Prefix = args.prefix
    def run(self):
        Cont_P = []
        #for profile in self.Cont_SI[self.start: min(self.Cont_SI.shape[0], self.start+self.stepi)]:
        for profile in self.Cont_SI[self.start: self.start+self.step]:
            pvalues = np.array([GetPermutationP(self.Cont_SI[:, i], profile[i])[
                               1] for i in range(profile.shape[0])])
            Cont_P.append(pvalues)
        Cont_P = np.array(Cont_P)
        np.save("{}_{}.npy".format(self.Prefix, self.start), Cont_P)


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input SI score Null data')
    parser.add_argument('-p', '--prefix', type=str, help='Prefix of output file')
    parser.add_argument('-s', '--start', type=int, help='start size')
    parser.add_argument('--step', default = 1000, type=int, help='step size')
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = scripts_Get_P_Matrix_From_Score(args)
    ins.run()

    return


if __name__ == '__main__':
    main()
