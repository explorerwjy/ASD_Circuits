# Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# MakeExpressionMat.py
#========================================================================================================

import argparse
import json
import numpy as np
import pandas as pd


class MakeExpressionMat:
    def __init__(self, args):
        # Note: Keys will be read as str
        self.Human2Mouse_Genes = json.load(open(args.human_genes))
        # Note: Keys will be read as str
        self.Mouse2Human_Genes = json.load(open(args.mouse_genes))
        self.STR_DF = pd.read_csv(args.structure_meta, index_col="id")
        self.DataDIR = args.exp_dir
        self.log2 = args.log
        self.OutFil = args.out
        print("Log:", self.log2, " OutFil:", self.OutFil)

    def run(self):
        All_Genes = []
        All_ExpEnergy = []
        for i, (entrez, v) in enumerate(self.Human2Mouse_Genes.items()):
            entrez = int(entrez)
            Symbol = v["symbol"]
            mouseHomo = v["mouseHomo"]
            g_All_Section_ID = []
            for m_symbol, m_entrez in mouseHomo:
                section_ids = self.Mouse2Human_Genes[str(
                    m_entrez)]["allen_section_data_set_id"]
                g_All_Section_ID.extend(section_ids)
            g_All_dat = []
            if len(g_All_Section_ID) == 0:
                continue
            for section in g_All_Section_ID:
                dat_df = pd.read_csv("{}/{}.csv".format(self.DataDIR, section))
                dat = []
                for str_id, row in self.STR_DF.iterrows():
                    _ = dat_df[dat_df["structure_id"] == str_id]
                    if len(_) > 0:
                        if self.log2:
                            #exp_energy = np.log2(1+_["expression_energy"].values[0])
                            exp_energy = _["expression_energy"].values[0]
                        else:
                            #exp_energy = np.log2(1+_["expression_energy"].values[0])
                            exp_energy = _["expression_energy"].values[0]
                    else:
                        exp_energy = np.nan
                    dat.append(exp_energy)
                g_All_dat.append(dat)
            g_All_dat = np.array(g_All_dat)
            g_avg_exp_energy = np.nanmean(g_All_dat, axis=0)
            All_Genes.append(entrez)
            All_ExpEnergy.append(g_avg_exp_energy)

            # if i > 10: # test purpuse
            #	break
        All_ExpEnergy = np.array(All_ExpEnergy)
        ExpMat = pd.DataFrame(
            data=All_ExpEnergy, index=All_Genes, columns=self.STR_DF["Name2"].values)
        ExpMat.to_csv(self.OutFil)


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human_genes', type=str, default='/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/human2mouse.0420.json',
                        help='JSON file contains human to mouse orthologs')
    parser.add_argument('--mouse_genes', type=str, default='/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/mouse2sectionID.0420.json',
                        help='JSON file contains mouse to human orthologs and allen sectionID Mapping')
    parser.add_argument('--structure_meta', default='/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/Selected_213_STRs.Meta.csv',
                        type=str, help='CSV file contains 213 Structure Used in this analysis')
    parser.add_argument('--exp_dir', default='/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/allen-mouse-brain-altas-str-unionzed/',
                        type=str, help='Directory contains allen mouse brain atlas str unionzed data')
    parser.add_argument('--log', type=bool, default=True,
                        help="log2 expression value or not")
    parser.add_argument('--out', type=str, required=True,
                        help="name of output expression matrix")
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = MakeExpressionMat(args)
    ins.run()

    return


if __name__ == '__main__':
    main()
