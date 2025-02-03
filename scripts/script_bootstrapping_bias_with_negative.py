#!/home/local/users/jw/anaconda2/bin/python
#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# script_bootstrapping_mutations.py
#========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/ASD_Circuits/src')
from ASD_Circuits import *


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol, allen_mouse_genes = LoadGeneINFO()

class script_bootstrapping_mutations:
	def __init__(self, args):
		self.Muts = pd.read_csv(args.variant)
		self.Nstart = args.nstart
		self.Nend = args.nend
		self.Match = args.match
		self.Dmis = args.Dmis
		self.DIR = args.location
		self.Mat = args.matrix
		self.Noneg = args.Noneg
	def run(self):
		ExpMat = pd.read_csv(self.Mat, index_col=0)
		directory = "/".join(self.DIR.split("/")[:-1])
		if not os.path.exists(directory):
			os.makedirs(directory)
		for i in range(self.Nstart, self.Nend+1, 1):
			print(i)
			bootstrapped_muts = self.Muts.sample(frac=1, replace=True)
			boot_gene2MutN = self.Mut2GeneDF(bootstrapped_muts, self.Dmis)
			ASD_Meta_Spec = AvgSTRZ_Weighted(ExpMat, boot_gene2MutN, Method = 1, NonNeg=False,  csv_fil = "{}/bootstrap.{}.csv".format(self.DIR, i))
			bootstrapped_muts.to_csv("{}/boot.muts.{}.csv".format(self.DIR, i))
			Circuit_STRs = ASD_Meta_Spec["STR"].values[:50]
			neg_genes = []
			for entrez in boot_gene2MutN.keys():
				if entrez not in ExpMat.index.values:
					continue
				g_biases = []
				for STR in Circuit_STRs:
					z_gene = ExpMat.loc[entrez, STR]
					g_biases.append(z_gene)
				g_avgBias = np.mean(g_biases)
				if g_avgBias <0:
					neg_genes.append(entrez)
			neg_gene_dict = {}
			for k,v in boot_gene2MutN.items():
				if k in neg_genes:
					neg_gene_dict[k] = v
			writer = csv.writer(open("{}/neg.boot.gene.{}.dict".format(self.DIR, i), 'wt'))
			for k,v in sorted(neg_gene_dict.items(), key=lambda x:x[1], reverse=True):
				writer.writerow([k,v])
			ASD_Neg_Spec = AvgSTRZ_Weighted(ExpMat, neg_gene_dict, Method = 1,csv_fil = "{}/ASD.Neg.Z2.{}.csv".format(self.DIR, i))

	def Mut2GeneDF(self, MutDF, Dmis, pLI=True):
		Select_Genes = np.array(list(set(MutDF["HGNC"].values)))
		dat = []
		gene2MutN = {}
		for g in Select_Genes:
			try:
				Entrez = int(GeneSymbol2Entrez[g])
			except:
				Entrez = -1
				continue
			Muts = MutDF[MutDF["HGNC"]==g]
			N_LGD, N_Mis, N_Dmis, N_Syn = CountMut(Muts)
			if pLI:
				try:
					pLI = float(Muts["ExACpLI"].values[0])
				except:
					pLI = 0.0
				if pLI >= 0.5:
					gene2MutN[Entrez] = N_LGD * 0.554 + N_Dmis * 0.333
				else:
					gene2MutN[Entrez] = N_LGD * 0.138 + N_Dmis * 0.130
			#if Dmis:
			#		gene2MutN[Entrez] = N_LGD * 1 + N_Dmis * 1
			else:
				gene2MutN[Entrez] = N_LGD * 0.357 + N_Dmis * 0.231
		return gene2MutN

def CountMut(DF, DmisSTR="REVEL", DmisCut=0.5):
	N_LGD, N_mis, N_Dmis, N_syn = 0,0,0,0
	for i, row in DF.iterrows():
		GeneEff = row["GeneEff"].split(";")[0]
		if GeneEff in ["frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_gained", "stop_lost"]:
			N_LGD += 1
		elif GeneEff == "missense":
			N_mis += 1
			if row["REVEL"] == ".":
				continue
			elif float(row["REVEL"]) >= 0.5:
				N_Dmis += 1
		elif GeneEff == "synonymous":
			N_syn += 1
	return N_LGD, N_mis, N_Dmis, N_syn

def GetOptions():
	parser = argparse.ArgumentParser()
	parser.add_argument('--nstart', type=int, required=True, help="Number of bootstrapping start time")
	parser.add_argument('--nend', type=int, required=True, help="Number of bootstrapping end time")
	parser.add_argument('-v','--variant', type=str, required=True, help = 'variant table used for bootstrap')
	parser.add_argument('-m', '--match', type=str, help = 'Matched gene table for Z-Z')
	parser.add_argument('-l', '--location', type=str, required=True, help = 'Dir and file name for saving restults')
	parser.add_argument('--Dmis', type=bool, default=1, help = 'Consider Dmis or not')
	parser.add_argument('--matrix', type=str, default="dat/z2.all.genes.str.mat.csv", help = 'Dir and file name for saving restults')
	parser.add_argument('--Noneg', type=bool, default=False, help = 'Whether Use Noneg bias, default=False')
	args = parser.parse_args()
	
	return args

def main():
	args = GetOptions()
	ins = script_bootstrapping_mutations(args)
	ins.run()

	return

if __name__=='__main__':
	main()
