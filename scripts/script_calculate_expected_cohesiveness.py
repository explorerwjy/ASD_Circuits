# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_calculate_expected_cohesiveness.py
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *
import pickle

class script_calculate_expected_cohesiveness:
    def __init__(self, args):
        self.size = args.size
        self.n_rand = args.n_rand
        self.graph_type = args.graph_type
        if self.graph_type == "complete":
            self.graph = LoadConnectome2()
        elif self.graph_type == "local":
            self.graph = LoadConnectome2("../dat/allen-mouse-conn/adj_mat_cartesian_local.csv")
        elif self.graph_type == "distal":
            self.graph = LoadConnectome2("../dat/allen-mouse-conn/adj_mat_cartesian_distal.csv")
        self.EdgeWeightsDict = EdgeDict(self.graph)
    def run(self):
        ALL_STRs = self.graph.vs["label"]
        STR_Cohe_Exp_Dict = {}
        circuit_size = self.size
        for i in range(self.n_rand):
            #print(ALL_STRs, circuit_size)
            circuit_strs = np.random.choice(ALL_STRs, circuit_size, replace=False)
            for v in self.graph.vs:
                if v['label'] in circuit_strs:
                    STR = v['label']
                    total_in = 0
                    total_all = 0
                    for v_neighbor in v.successors():
                        weight = self.EdgeWeightsDict["{}-{}".format(v['label'], v_neighbor["label"])]
                        if v_neighbor['label'] in circuit_strs:
                            total_in += weight
                        total_all += weight
                    try:
                        cohe = total_in/total_all
                    except:
                        cohe = 0
                    if STR not in STR_Cohe_Exp_Dict:
                        STR_Cohe_Exp_Dict[STR] = []
                    STR_Cohe_Exp_Dict[STR].append(cohe)
        f = open("/home/jw3514/Work/ASD_Circuits/dat/Circuits/Cohe_Exp/{}.weighted.size.{}.pkl".format(self.graph_type, self.size), 'wb')
        pickle.dump(STR_Cohe_Exp_Dict, f) 
        f.close()

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, help='size of the circuits')
    parser.add_argument('-n', '--n_rand', type=int, default=10000, help='number of randomization')
    parser.add_argument('--graph_type', type=str, default='complete', help='what graph file to use')
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = script_calculate_expected_cohesiveness(args)
    ins.run()
    return


if __name__ == '__main__':
    main()
