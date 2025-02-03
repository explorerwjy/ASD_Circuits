import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *
import argparse

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bias', type=str, help="Mutation Bias DF")
    parser.add_argument('-o', '--outdir', type=str, help="Output dir")
    args = parser.parse_args()
    return args

def main():

    #OutDIR = "/home/jw3514/Work/ASD_Circuits/dat/Circuits/SA/biaslims2/"
    #BiasDF = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv", index_col="STR")
    args = GetOptions()
    BiasDF = pd.read_csv(args.bias, index_col=0)
    OutDIR = args.outdir + "/"

    sizes = np.arange(10, 100, 1)
    for i, t in enumerate(sizes):
        fout = open(OutDIR + "biaslim.size.{}.txt".format(t), 'w')
        writer = csv.writer(fout)
        lims = BiasLim(BiasDF, t)
        for size, bias in lims:
            writer.writerow([size, bias])
        fout.close()

main()
