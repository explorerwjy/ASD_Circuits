import pandas as pd
import csv 
import re
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
import random
import bisect
import collections
import scipy.stats 
import re
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import halfcauchy
from scipy.integrate import quad, dblquad
import itertools
import mygene

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)] 
# Files
protein_coding_gene_file = "/Users/jiayao/Work/Resources/protein-coding_gene.txt"
wigler_predicted_lgd = "/Users/jiayao/Work/BrainDisorders/data/functions/wigler-predicted-lgd.txt"
wigler_predicted_male = "/Users/jiayao/Work/BrainDisorders/data/functions/wigler-predicted-male.txt"
wigler_predicted_fem = "/Users/jiayao/Work/BrainDisorders/data/functions/wigler-predicted-fem.txt"
wigler_fam_info = "/Users/jiayao/Work/BrainDisorders/data/nature13908-s2/Supplementary_Table_1.xlsx" #IQ in it
ASD_high_conf = "/Users/jiayao/Work/BrainDisorders/data/ASD_high_confidence_genes_post80.txt"
# Entrez ID of genes in each category; No overlap
vitkup_psd = "/Users/jiayao/Work/BrainDisorders/data/functions/asd-vitkup-snv-cnv-psd.txt"
vitkup_sig_skel = "/Users/jiayao/Work/BrainDisorders/data/functions/asd-vitkup-snv-cnv-sig-skel.txt"
vitkup_channel = "/Users/jiayao/Work/BrainDisorders/data/functions/asd-vitkup-snv-cnv-channel.txt"
vitkup_chromatin = "/Users/jiayao/Work/BrainDisorders/data/functions/asd-vitkup-snv-cnv-chromatin.txt"

go_psd = "/Users/jiayao/Work/BrainDisorders/data/functions/go_synapse.list"
go_sig_skel = "/Users/jiayao/Work/BrainDisorders/data/functions/go_cyto_sig.list"
go_channel = "/Users/jiayao/Work/BrainDisorders/data/functions/go_channel.list"
go_chromatin = "/Users/jiayao/Work/BrainDisorders/data/functions/go_chromatin.list"

andy_psd_channel = "/Users/jiayao/Work/BrainDisorders/JW/Functional_Cluster_Gene_Lists/CLUST_NEURO_PSD_CHANN.txt"
andy_cytp_sign = "/Users/jiayao/Work/BrainDisorders/JW/Functional_Cluster_Gene_Lists/CLUST_NEURO_CYTO_SIGN.txt"
andy_snf_bromo = "/Users/jiayao/Work/BrainDisorders/JW/Functional_Cluster_Gene_Lists/CLUST_DEVEL_SNF_BROMO.txt"
andy_znf = "/Users/jiayao/Work/BrainDisorders/JW/Functional_Cluster_Gene_Lists/CLUST_DEVEL_ZN_FINGER.txt"


Stages = ["2A", "2B", "3A", "3B"] + [str(x) for x in range(4,12)]

VEP_LGD = ['frameshift_variant',
 'frameshift_variant,splice_region_variant',
 'splice_acceptor_variant',
 'splice_donor_variant',
 'splice_donor_variant,coding_sequence_variant',
 'splice_donor_variant,coding_sequence_variant,intron_variant',
 'start_lost',
 'stop_gained',
 'stop_gained,frameshift_variant',
 'stop_gained,splice_region_variant',
 'stop_lost',
 'stop_lost,inframe_deletion']

class H2Analysis:
    def __init__(self):
        pass
    def PlotQQ(P, N=None):
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15}
        mpl.rc('font', **font)
        P = np.array(P) 
        P.sort(reverse=True)
        Q = [-1*math.log(x,10) for x in P]
        top = int(Q[-1]) + 1
        
        N = len(P) if N == None else N
        P_null = [float(i+1)/N for i in xrange(N)]
        P_null.sort(reverse=True)
        Q_null = [-1*math.log(x,10) for x in P_null]
        colors = (0,0,0)
        plt.scatter(Q_null, Q, c=colors, alpha=0.5)
        plt.plot([0, top], [0, top], ls="-")
        plt.xlabel('Exp Q')
        plt.ylabel('Obs Q')
        plt.show()
    def _gen_data(fhs, columns, sep):
        """
        iterate over the files and yield chr, start, pvalue
        """
        for fh in fhs:
            for line in fh:
                if line[0] == "#": continue
                toks = line.split(sep)
                yield toks[columns[0]], int(toks[columns[1]]), float(toks[columns[2]])
    def chr_cmp(a, b):
        a = a.lower().replace("_", ""); b = b.lower().replace("_", "")
        achr = a[3:] if a.startswith("chr") else a
        bchr = b[3:] if b.startswith("chr") else b

        try:
            return cmp(int(achr), int(bchr))
        except ValueError:
            if achr.isdigit() and not bchr.isdigit(): return -1
            if bchr.isdigit() and not achr.isdigit(): return 1
            # X Y
            return cmp(achr, bchr)
    def chr_loc_cmp(alocs, blocs):
        return chr_cmp(alocs[0], blocs[0]) or cmp(alocs[1], blocs[1])
    def manhattan(fhs, columns, image_path, no_log, colors, sep, title, lines, ymax):
        xs = []
        ys = []
        cs = []
        colors = cycle(colors)
        xs_by_chr = {}

        last_x = 0
        data = sorted(_gen_data(fhs, columns, sep), cmp=chr_loc_cmp)

        for seqid, rlist in groupby(data, key=itemgetter(0)):
            color = colors.next()
            rlist = list(rlist)
            region_xs = [last_x + r[1] for r in rlist]
            xs.extend(region_xs)
            ys.extend([r[2] for r in rlist])
            cs.extend([color] * len(rlist))

            xs_by_chr[seqid] = (region_xs[0] + region_xs[-1]) / 2

            # keep track so that chrs don't overlap.
            last_x = xs[-1]

        xs_by_chr = [(k, xs_by_chr[k]) for k in sorted(xs_by_chr.keys(), cmp=chr_cmp)]

        xs = np.array(xs)
        ys = np.array(ys) if no_log else -np.log10(ys)

        plt.close()
        f = plt.figure()
        ax = f.add_axes((0.1, 0.09, 0.88, 0.85))

        if title is not None:
            plt.title(title)

        ax.set_ylabel('-log10(p-value)')
        if lines:
            ax.vlines(xs, 0, ys, colors=cs, alpha=0.5)
        else:
            ax.scatter(xs, ys, s=2, c=cs, alpha=0.8, edgecolors='none')

        # plot 0.05 line after multiple testing.
        ax.axhline(y=-np.log10(0.05 / len(data)), color='0.5', linewidth=2)
        plt.axis('tight')
        plt.xlim(0, xs[-1])
        plt.ylim(ymin=0)
        if ymax is not None: plt.ylim(ymax=ymax)
        plt.xticks([c[1] for c in xs_by_chr], [c[0] for c in xs_by_chr], rotation=-90, size=8.5)
        print (sys.stderr, "saving to: %s" % image_path)
        plt.savefig(image_path)
        #plt.show() 

def get_gene_entrez_symbol_map():
    res = {}
    reader = csv.reader(open(protein_coding_gene_file), delimiter="\t")
    header = reader.next()
    idx_symbol, idx_entrez_id = header.index("symbol"), header.index("entrez_id")
    for row in reader:
        res[row[idx_entrez_id]] = row[idx_symbol]
    return res
class DevStageExp:
    def __init__(self, DevStage):
        self.DevStage = DevStage
        self.exps = np.array([])
        self.mean_exp = 0
        self.median = 0
    def add_exp(self, value):
        self.exps = np.append(self.exps, float(value))
        #self.mean_exp = round(np.mean(self.exps), 4)
    def get_nonzero_mean(self):
        try:
            #self.non_zero_exps = [math.log(x, 10) for x in self.exps if x]
            self.non_zero_exps = [x for x in self.exps if x]
            N = len(self.non_zero_exps)
            return np.mean(self.non_zero_exps), math.sqrt(np.var(self.non_zero_exps))/math.sqrt(N), np.median(self.non_zero_exps)
        except ZeroDivisionError:
            return 0, 0, 0
class BrainSpan:
    def __init__(self):
        # store all exp value at a developemt time point
        #Stages = map(str,range(1,16))
        #self.Stages = ["1", "2A", "2B", "3A", "3B"] + map(str,range(4,12))
        self.Stages = ["2A", "2B", "3A", "3B"] + [str(x) for x in range(4,12)]
        self.Descriptions = ["Early fetal", "Early fetal", "Early mid-fetal", "Early mid-fetal", "Late mid-fetal", "Late fetal", "Early infancy", "Late infancy", "Early childhood", "Late childhood", "Adolescence", "Adulthood"]
        self.Regions = []
        self.Regionsgt20 = ['OFC', 'VFC', 'HIP', 'ITC', 'AMY', 'DFC', 'STC', 'MFC', 'STR', 'IPC', 'V1C', 'S1C', 'A1C', 'M1C', 'CBC', 'MD']
        self.TimeDict = dict(zip(["8 pcw", "9 pcw", "12 pcw", "13 pcw", "16 pcw", "17 pcw", "19 pcw", "21 pcw", "24 pcw", "25 pcw", "26 pcw", "35 pcw", "37 pcw", "4 mos", "10 mos", "1 yrs", "2 yrs", "3 yrs", "4 yrs", "8 yrs", "11 yrs", "13 yrs", "15 yrs", "18 yrs", "19 yrs", "21 yrs", "23 yrs", "30 yrs", "36 yrs", "37 yrs", "40 yrs"], [("2A","2", "Early prenatal"), ("2A","2", "Early prenatal"), ("2B", "3", "Early prenatal"), ("3A", "4", "Early mid-prenatal"), ("3B", "5", "Early mid-prenatal"), ("3B", "5", "Early mid-prenatal"), ("4", "6", "Late mid-prenatal"), ("4", "6", "Late mid-prenatal"), ("4", "6", "Late mid-prenatal"), ("5", "7", "Late prenatal"), ("5", "7", "Late prenatal"), ("5", "7", "Late prenatal"), ("5", "7", "Late prenatal"), ("6", "8", "Early infancy"), ("7", "9", "Late infancy"), ("7", "9", "Late infancy"), ("8", "10", "Early childhood"), ("8", "10", "Early childhood"), ("8", "10", "Early childhood"), ("9", "11", "Late childhood"), ("9", "11", "Late childhood"), ("10", "12", "Adolescence"),("10", "12", "Adolescence"),("10", "12", "Adolescence"),("10", "12", "Adolescence"), ("11", "14", "adulthood"), ("11", "14", "adulthood"), ("11", "14", "adulthood"), ("11", "14", "adulthood"), ("11", "14", "adulthood"), ("11", "14", "adulthood")]))
    def TemporalMap2(self, age):
        num, unit = age.split()
        num = float(num)
        if unit == 'pcw':
            if num >= 4 and num < 8:
                return ("1","1", "Embryonic")
            elif num >= 8 and num <10:
                return ("2A","2", "Early prenatal")
            elif num >= 10 and num <13:
                return ("2B", "3", "Early prenatal")
            elif num >= 13 and num <16:
                return ("3A", "4", "Early mid-prenatal")
            elif num >= 16 and num <19:
                return ("3B", "5", "Early mid-prenatal")
            elif num >= 19 and num <24:
                return ("4", "6", "Late mid-prenatal")
            elif num >= 24 and num <38:
                return ("5", "7", "Late prenatal")
            else:
                print ("Unexpected Value")
        elif unit == 'mos':
            if num>= 0 and num < 6:
                return ("6", "8", "Early infancy")
            elif num>= 6 and num < 12:
                return ("7", "9", "Late infancy")
        elif unit == 'yrs':
            if num >= 1 and num < 6:
                return ("8", "10", "Early childhood")
            elif num >= 6 and num < 12:
                return ("9", "11", "Late childhood")
            elif num >= 12 and num < 20:
                return ("10", "12", "Adolescence")
            elif num >= 20 and num < 40:
                return ("11", "13", "Young adulthood")
            elif num >= 40 and num < 60:
                return ("11", "14", "Young adulthood")
            elif num >= 60:
                return ("11", "15", "Young adulthood")
    def TemporalMap(self, age):
        return self.TimeDict[age]
    # Return res1: sequence of exp in 12 dev stages res2: num of samples at each timepoint
    def DisplayStageExp(self, Dict, out="nonzero"):
        res1 = []
        res2 = []
        res3 = []
        res4 = []
        for stage in self.Stages:
            try:
                #res1.append(Dict[stage].mean_exp) # Exp avg
                stage, Dict[stage].get_nonzero_mean()
                if out == "all":
                    res1.append(Dict[stage].mean_exp)
                else:
                    mean, std, median = Dict[stage].get_nonzero_mean()
                    res1.append(mean) # Exp avg
                    res3.append(std)
                    res4.append(median)
                res2.append(len(Dict[stage].exps)) # Num of samples at that time point
            except KeyError:
                res1.append(0)
                res2.append(0)
                res3.append(0)
                res4.append(0)
        return res1, res2, res3, res4
    def Look(self, Gene, structure_acronym, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        GeneExonExp = ExonExp[ExonExp[0].isin(list(Exons["row_num"]))]
        ID_start, ID_end = list(Exons["row_num"])[0], list(Exons["row_num"])[-1]
        region_exon_meta = bp_exon_col_meta[bp_exon_col_meta["structure_acronym"]==structure_acronym]
        plt.close('all')
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : fontsize}
        mpl.rc('font', **font)
        res = {}
        for exon_id in xrange(ID_start,ID_end+1):
            stages = {}
            for index,row in region_exon_meta.iterrows():
                Period = row["Period"]
                if Period not in stages:
                    stages[Period] = DevStageExp(Period)
                stages[Period].add_exp(GeneExonExp.get_value(exon_id-1, row["column_num"]))
            seq, Nsamples = DisplayStageExp(stages)
            if drop_low_exp:
                if seq.count(0) >= 4:
                    res[exon_id] = None
                    continue
            if smooth:
                res[exon_id] = self.smooth_func(seq)
            else:
                res[exon_id] = seq 
        fig, ax = plt.subplots(dpi=200)
        plt.title("Gene:{} Region:{}".format(Gene, structure_acronym))
        #largest = 0
        for exon_id in xrange(ID_start,ID_end+1):
            if res[exon_id] == None:
                continue
            #if min(np.array([math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])) < -2:
            #if (np.array([math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])[7]) > largest:
            #    largest = (np.array([math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])[7])
            #    largest_id = exon_id
            ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])
        #print exon_id
        ax.grid(True)
        ax.axvline(x=7.5)
        #plt.ylim(-5, 2)
        #Stages = Stages + [0,0,0,0,0]
        plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        plt.show()
    # I forget what it does
    def Look2(self, Gene, structure_acronym, bp_exon_row_meta, bp_exon_col_meta, ExonExp, fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        GeneExonExp = ExonExp[ExonExp[0].isin(list(Exons["row_num"]))]
        ID_start, ID_end = list(Exons["row_num"])[0], list(Exons["row_num"])[-1]
        region_exon_meta = bp_exon_col_meta[bp_exon_col_meta["structure_acronym"]==structure_acronym]
        plt.close('all')
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : fontsize}
        mpl.rc('font', **font)
        res = {}
        for exon_id in xrange(ID_start,ID_end+1):
            stages = {}
            for index,row in region_exon_meta.iterrows():
                Period = row["Period"]
                if Period not in stages:
                    stages[Period] = DevStageExp(Period)
                stages[Period].add_exp(GeneExonExp.get_value(exon_id-1, row["column_num"]))
            res[exon_id] = DisplayStageExp(stages)
        fig, ax = plt.subplots(dpi=200)
        plt.title("Gene:{} Region:{}".format(Gene, structure_acronym))
        for exon_id in xrange(ID_start,ID_end+1):
            #if np.array([math.log(x, 2) if x!=0 else 0 for x in res[exon_id]]).
            #ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])
            ax.plot(range(2,14),res[exon_id])
        ax.grid(True)
        ax.axvline(x=7.6)
        #Stages = Stages + [0,0,0,0,0]
        plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        plt.show()
    def example_plot(self, ax, structure_acronym, fontsize=6):
        ax.locator_params(nbins=3)
        #ax.set_xlabel('Dev Stages', fontsize=fontsize)
        ax.set_xlabel('Dev Stages', fontsize=fontsize)
        ax.set_ylabel('log2 fold change', fontsize=fontsize)
        ax.set_title(structure_acronym, fontsize=fontsize)
    # Display exon express in a series of regions, for a certain gene
    def LookGrid(self, Gene, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        GeneExonExp = ExonExp[ExonExp[0].isin(list(Exons["row_num"]))]
        ID_start, ID_end = list(Exons["row_num"])[0], list(Exons["row_num"])[-1]
        NN = int(math.sqrt(len(structure_acronyms))) 
        #print len(structure_acronyms), NN
        if NN * NN == len(structure_acronyms):
            XX, YY = NN, NN
        else:
            XX, YY = NN+1 , NN + 1
        plt.close('all')
        fig = plt.figure(dpi=800)
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : fontsize}
        mpl.rc('font', **font)
        from mpl_toolkits.axes_grid1 import Grid
        grid = Grid(fig, rect=111, nrows_ncols=(XX,YY), axes_pad=0.25, label_mode='L',)
        #print grid
        for i, structure_acronym in enumerate(structure_acronyms):
            sys.stdout.write("\r{}".format(i+1))
            region_exon_meta = bp_exon_col_meta[bp_exon_col_meta["structure_acronym"]==structure_acronym]
            res = {}
            for exon_id in xrange(ID_start,ID_end+1):
                stages = {}
                for index,row in region_exon_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    stages[Period].add_exp(GeneExonExp.get_value(exon_id-1, row["column_num"]))
                seq, Nsamples = DisplayStageExp(stages)
                if drop_low_exp:
                    if seq.count(0) >= 4:
                        res[exon_id] = None
                        continue
                res[exon_id] = smooth_func(seq) if smooth else seq
                #if smooth:
                #    #print DisplayStageExp(stages)[0]
                #    res[exon_id] = smooth_func(seq)
                #    #print res[exon_id]
                #else:
                #    res[exon_id] = seq
            ax = grid[i]
            example_plot(ax, structure_acronym)
            for exon_id in xrange(ID_start,ID_end+1):
                if res[exon_id] == None:
                    continue
                #ax.plot(range(2,14),res[exon_id])
                #ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])
                ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])
            ax.grid(True)
            ax.axvline(x=7.5)
            #plt.ylim(-2, 2)
            #plt.xticks(np.arange(2,14,2), Stages, rotation=20)
            #ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        plt.tight_layout(pad=4, w_pad=5, h_pad=10)
        #plt.xticks(np.arange(2,14,2), Stages, rotation=20)
        #plt.ylim(-2, 2)
        sys.stdout.write("\rplotting....")
        plt.show()
    def smooth_func(self, seq):
        seq.insert(0,0)
        seq.append(0)
        new = []
        i = 1
        while i < len(seq) - 1:
            nonzero = [ x for x in seq[i-1: i+2] if x ]
            try:
                new.append(sum(nonzero)/len(nonzero))
            except ZeroDivisionError:
                #print "Three 0 data at stage",i
                new.append(0)
            i += 1
        return new
    def format_func(self, value, tick_number):
        try:
            return self.Stages[int(value)-2]
        except:
            #print value
            return 0
    class expvalue:
        def __init__(self):
            self.value = []
        def addvalue(self, value):
            self.value.append(value)
            self.mean = self.value.mean()
    def LookGridSumRegion(self, Gene, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        GeneExonExp = ExonExp[ExonExp[0].isin(list(Exons["row_num"]))]
        ID_start, ID_end = list(Exons["row_num"])[0], list(Exons["row_num"])[-1]
        plt.close('all')
        fig = plt.figure(dpi=800)
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : fontsize}
        mpl.rc('font', **font)
        from mpl_toolkits.axes_grid1 import Grid
        #grid = Grid(fig, rect=111, nrows_ncols=(XX,YY), axes_pad=0.25, label_mode='L',)
        res = {}
        for exon_id in xrange(ID_start, ID_end+1):
            stages = {}
            for j, structure_acronym in enumerate(structure_acronyms):
                region_exon_meta = bp_exon_col_meta[bp_exon_col_meta["structure_acronym"]==structure_acronym]
                for index, row in region_exon_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    stages[Period].add_exp(GeneExonExp.get_value(exon_id-1, row["column_num"]))
            #res[exon_id]
            seq = self.DisplayStageExp(stages)[0]
            if drop_low_exp:
                if seq.count(0) >= 4:
                    res[exon_id] = None
                    continue
            if smooth:
                res[exon_id] = self.smooth_func(seq)
            else:
                res[exon_id] = seq

        fig, ax = plt.subplots(dpi=200)
        plt.title("Gene:{} over Region:{}".format(Gene, ",".join(structure_acronyms)))
        for exon_id in xrange(ID_start,ID_end+1):
            if res[exon_id] == None:
                continue
            #if np.array([math.log(x, 2) if x!=0 else 0 for x in res[exon_id]]).
            ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])
            #ax.plot(range(2,14),res[exon_id])
            #print len(res[exon_id]), res[exon_id]
            ax.plot(range(2,14),res[exon_id])
        ax.grid(True)
        ax.axvline(x=7.5)
        #Stages = Stages + [0,0,0,0,0]
        plt.xticks(np.arange(2,14), self.Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        plt.show()
    def LookGeneExonSumOverRegionwithTargetUntarget(self, Gene, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, GeneDat=None, smooth=True, drop_low_exp=True, space = 'log', fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        TargetedExons = Exons[Exons["Vars"]!=""]
        UnTargetedExons = Exons[Exons["Vars"]==""]
        #print TargetedExons
        plt.close('all')
        fig, ax = plt.subplots(dpi=100)
        plt.title("Gene:{} over Region:{}".format(Gene, ",".join(structure_acronyms)))
        res = {}
        
        for exon_id in list(TargetedExons["row_num"]):
            exon_id = int(exon_id)
            seq, error_var, median = self.LoadingDat2SeqCrossRecordCrossRegion([exon_id], structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, space = space,  smooth=True)
            ax.plot(range(2,14), seq, label=str(exon_id))
        for exon_id in list(UnTargetedExons["row_num"]):
            exon_id = int(exon_id)
            seq, error_var, median = self.LoadingDat2SeqCrossRecordCrossRegion([exon_id], structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, space = space, shutup=True)
            ax.plot(range(2,14), seq, '--', alpha = 0.3, color='grey') 
        if GeneDat != None:# Plot Gene exp over stages 
            stages = {}
            GeneExp, GeneRow, GeneCol = GeneDat
            Genes = GeneRow[GeneRow["gene_symbol"]==Gene]
            Gene = GeneExp[GeneExp[0].isin(list(Genes["row_num"]))]
            gene_ids = list(Genes["row_num"])
            for gene_id in gene_ids:
                for j, structure_acronym in enumerate(structure_acronyms): # sum over structures/regions
                    region_meta = GeneCol[GeneCol["structure_acronym"]==structure_acronym]
                    for index, row in region_meta.iterrows():
                        Period = row["Period"]
                        if Period not in stages:
                            stages[Period] = DevStageExp(Period)
                        exp = GeneExp.get_value(gene_id-1, row["column_num"])
                        if space == 'log':
                            exp = math.log(exp+1, 2)
                        stages[Period].add_exp(exp)
                seq, Nsample, median, var = self.DisplayStageExp(stages)
                #print seq
                if smooth:
                    res[gene_id] = self.smooth_func(seq)
                else:
                    res[gene_id] = seq
            #add_layout(LinearAxis(y_range_name="GeneRPKM"), 'right')
            #ax2 = ax.twinx()
            for gene_id in gene_ids:
                #print res[gene_id]
                #ax2.plot(range(2,14), [math.log(x,2) if x !=0 else 0 for x in res[gene_id]], color="black") 
                #ax2.plot(range(2,14), res[gene_id], color="black") 
                ax.plot(range(2,14), res[gene_id], color="black") 
            #ax2.set_ylabel("GeneRPKM")
        ax.grid(True)
        ax.axvline(x=7.5)
        #Stages = Stages + [0,0,0,0,0]
        plt.xticks(np.arange(2,14), self.Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        plt.legend()
        ax.legend()
        plt.show()
    def NoiseEstimate():
        pass
    def AssignVar2Exon(self, bp_exon_row_meta, VarFile):
        reader = csv.reader(open(VarFile, 'rb'))
        header = reader.next()
        res = {} # k:gene v:list of variants in that gene
        for row in reader:
            tmp = dict(zip(header, row))
            if tmp["Gene"] not in res:
                res[tmp["Gene"]] = [tmp]
            else:
                res[tmp["Gene"]].append(tmp)
        bp_exon_row_meta["Vars"] = ""
        for i, row in bp_exon_row_meta.iterrows():
            sys.stdout.write("\r{}".format(i))
            start, end = int(row["start"]), int(row["end"])
            if row["gene_symbol"] not in res:
                continue
            for var in res[row["gene_symbol"]]:
                pos = int(var["Position"])
                if pos >= start and pos <= end:
                    ##bp_exon_row_meta["Vars"].append(var["cDnaVariant"])
                    row["Vars"] = row["Vars"] + ";" + var["cDnaVariant"] 
                    bp_exon_row_meta.at[i, "Vars"] = bp_exon_row_meta.get_value(i, "Vars") + ";" + var["cDnaVariant"] 
                    continue
    # Assign Category to a gene
    # Cates are dict of category:geneset
    def AssignFunc(self, gene, Cates): # Assign Category to each gene
        res = []
        for k,v in Cates.items():
            if gene in v:
                res.append(k)
        return ";".join(res)
    # Assign Variants to row meta data file of Exons. 
    # Markers on 1)Male vs. Female; 2)IQ70+ vs. IQ70- 3)Functions
    def AssignVar2Exon2(self, bp_exon_row_meta, VarFile, IntersectionWithPredicted = True, ProSib="Pro"):
        VarDF = pd.read_excel(VarFile)
        # Load annotations to gene
        entrez_symbol_map = get_gene_entrez_symbol_map()
        wigler_predicted_lgd_genes = set([entrez_symbol_map[x.strip()] for x in file(wigler_predicted_lgd)])
        #vitkup_channel_genes = set([entrez_symbol_map[x.strip()] for x in file(vitkup_channel)])
        #vitkup_chromatin_genes = set([entrez_symbol_map[x.strip()] for x in file(vitkup_chromatin)])
        #vitkup_psd_genes = set([entrez_symbol_map[x.strip()] for x in file(vitkup_psd)])
        #vitkup_sig_skel_genes = set([entrez_symbol_map[x.strip()] for x in file(vitkup_sig_skel)])
        channel_genes = set([entrez_symbol_map.get(x.strip(), None) for x in file(go_channel)])
        chromatin_genes = set([entrez_symbol_map.get(x.strip(), None) for x in file(go_chromatin)])
        psd_genes = set([entrez_symbol_map.get(x.strip(), None) for x in file(go_psd)])
        sig_skel_genes = set([entrez_symbol_map.get(x.strip(), None) for x in file(go_sig_skel)])
        fam_info = pd.read_excel(wigler_fam_info)
        famID2NVIQ = dict(zip(list(fam_info["familyId"]), list(fam_info["probandNVIQ"])))
        famID2VIQ = dict(zip(list(fam_info["familyId"]), list(fam_info["probandVIQ"])))
        
        Cates = dict(zip(["channel", "chromatin", "psd", "sig_skel"],[channel_genes,chromatin_genes,psd_genes,sig_skel_genes]))
        res = {} # k:gene v:list of variants in that gene
        var_genders = {}
        var_IQ = {}
        for i, row in VarDF.iterrows():
            if IntersectionWithPredicted:
                if row["effectGene"] not in wigler_predicted_lgd_genes:
                    continue
            # Get Variant Info
            gene,var = row["effectGene"], row["location"]
            if gene not in res:
                res[gene] = [row["location"]]
            else:
                res[gene].append(row["location"])
            # Get Gender Info
            if var not in var_genders:
                if ProSib == 'Pro':
                    var_genders[var] = row["inChild"][:2]
                else:
                    var_genders[var] = row["inChild"][-2:]
                #var_genders[var] = [row["inChild"][:2]]
            else:
                if ProSib == 'Pro':
                    var_genders[var] = var_genders[var] + ";" + row["inChild"][:2]
                else:
                    var_genders[var] = var_genders[var] + ";" + row["inChild"][-2:]
            # Get IQ Info
            if var not in var_IQ:
                FamID = row["familyId"]
                var_IQ[var] = (famID2VIQ[FamID],famID2NVIQ[FamID])
        bp_exon_row_meta_rep = bp_exon_row_meta.copy(deep=True)
        bp_exon_row_meta_rep["NVIQ70"] = ""
        bp_exon_row_meta_rep["VIQ70"] = ""
        bp_exon_row_meta_rep["Gender"] = ""
        bp_exon_row_meta_rep["Func"] = ""
        bp_exon_row_meta_rep["Vars"] = ""
        bp_exon_row_meta_rep["GeneHited"] = "F"
        bp_exon_row_meta_rep["Last"] = "F"
        LAST_GENE = None
        for i, row in bp_exon_row_meta_rep.iterrows():
            sys.stdout.write("\r{}".format(i))
            gene = row["gene_symbol"]
            if gene != LAST_GENE and gene != None:
                bp_exon_row_meta_rep.at[i-1, "Last"] = "T" 
            LAST_GENE = gene
            if gene not in res:
                continue
            bp_exon_row_meta_rep.at[i, "GeneHited"] = "T"
            start, end = int(row["start"]) - 2 , int(row["end"]) + 2
            gene = row["gene_symbol"]
            # Assign Variant to Exon
            for var in res[gene]:
                pos = int(var.split(":")[1])
                #print pos
                if pos >= start and pos <= end: #Var In This Exon
                    bp_exon_row_meta_rep.at[i, "Vars"] = bp_exon_row_meta_rep.get_value(i, "Vars") + ";" + var if bp_exon_row_meta_rep.get_value(i, "Vars") != "" else var
                    bp_exon_row_meta_rep.at[i, "VIQ70"] = var_IQ[var][0]
                    bp_exon_row_meta_rep.at[i, "NVIQ70"] = var_IQ[var][1]
                    bp_exon_row_meta_rep.at[i, "Gender"] = bp_exon_row_meta_rep.get_value(i, "Gender") + ";" + var_genders[var] if bp_exon_row_meta_rep.get_value(i, "Gender") != "" else var_genders[var]
                    continue
            # Assign Other Info
            bp_exon_row_meta_rep.at[i, "Func"] = self.AssignFunc(gene, Cates)
        bp_exon_row_meta_rep.at[i, "Last"] = "T" #Last Exon in DF is last exon of that gene
        return bp_exon_row_meta_rep
    def AssignVar2Exon3(self, bp_exon_row_meta, VarFile, IntersectionWithPredicted = True, ProSib="Pro"):
        VarDF = pd.read_excel(VarFile)
        # Load annotations to gene
        entrez_symbol_map = get_gene_entrez_symbol_map()
        wigler_predicted_lgd_genes = set([entrez_symbol_map[x.strip()] for x in file(wigler_predicted_lgd)])
        channel_psd_genes = set([x.strip() for x in file(andy_psd_channel)])
        chromatin_genes = set([x.strip() for x in file(andy_snf_bromo)])
        cyto_sign = set([x.strip() for x in file(andy_cytp_sign)])
        tf_rbp = set([x.strip() for x in file(andy_znf)])
        fam_info = pd.read_excel(wigler_fam_info)
        famID2NVIQ = dict(zip(list(fam_info["familyId"]), list(fam_info["probandNVIQ"])))
        famID2VIQ = dict(zip(list(fam_info["familyId"]), list(fam_info["probandVIQ"])))
        
        Cates = dict(zip(["Channel&PSD", "Chromatin", "Cytp&Sig", "TF&RBP"],[channel_psd_genes,chromatin_genes,cyto_sign,tf_rbp]))
        res = {} # k:gene v:list of variants in that gene
        var_genders = {}
        var_IQ = {}
        for i, row in VarDF.iterrows():
            if IntersectionWithPredicted:
                if row["effectGene"] not in wigler_predicted_lgd_genes:
                    continue
            # Get Variant Info
            gene,var = row["effectGene"], row["location"]
            if gene not in res:
                res[gene] = [row["location"]]
            else:
                res[gene].append(row["location"])
            # Get Gender Info
            if var not in var_genders:
                if ProSib == 'Pro':
                    var_genders[var] = row["inChild"][:2]
                else:
                    var_genders[var] = row["inChild"][-2:]
                #var_genders[var] = [row["inChild"][:2]]
            else:
                if ProSib == 'Pro':
                    var_genders[var] = var_genders[var] + ";" + row["inChild"][:2]
                else:
                    var_genders[var] = var_genders[var] + ";" + row["inChild"][-2:]
            # Get IQ Info
            if var not in var_IQ:
                FamID = row["familyId"]
                var_IQ[var] = (famID2VIQ[FamID],famID2NVIQ[FamID])
        bp_exon_row_meta_rep = bp_exon_row_meta.copy(deep=True)
        bp_exon_row_meta_rep["NVIQ70"] = ""
        bp_exon_row_meta_rep["VIQ70"] = ""
        bp_exon_row_meta_rep["Gender"] = ""
        bp_exon_row_meta_rep["Func"] = ""
        bp_exon_row_meta_rep["Vars"] = ""
        bp_exon_row_meta_rep["GeneHited"] = "F"
        bp_exon_row_meta_rep["Last"] = "F"
        LAST_GENE = None
        for i, row in bp_exon_row_meta_rep.iterrows():
            sys.stdout.write("\r{}".format(i))
            gene = row["gene_symbol"]
            if gene != LAST_GENE and gene != None:
                bp_exon_row_meta_rep.at[i-1, "Last"] = "T" 
            LAST_GENE = gene
            if gene not in res:
                continue
            bp_exon_row_meta_rep.at[i, "GeneHited"] = "T"
            start, end = int(row["start"]) - 2 , int(row["end"]) + 2
            gene = row["gene_symbol"]
            # Assign Variant to Exon
            for var in res[gene]:
                pos = int(var.split(":")[1])
                #print pos
                if pos >= start and pos <= end: #Var In This Exon
                    bp_exon_row_meta_rep.at[i, "Vars"] = bp_exon_row_meta_rep.get_value(i, "Vars") + ";" + var if bp_exon_row_meta_rep.get_value(i, "Vars") != "" else var
                    bp_exon_row_meta_rep.at[i, "VIQ70"] = var_IQ[var][0]
                    bp_exon_row_meta_rep.at[i, "NVIQ70"] = var_IQ[var][1]
                    bp_exon_row_meta_rep.at[i, "Gender"] = bp_exon_row_meta_rep.get_value(i, "Gender") + ";" + var_genders[var] if bp_exon_row_meta_rep.get_value(i, "Gender") != "" else var_genders[var]
                    continue
            # Assign Other Info
            bp_exon_row_meta_rep.at[i, "Func"] = self.AssignFunc(gene, Cates)
        bp_exon_row_meta_rep.at[i, "Last"] = "T" #Last Exon in DF is last exon of that gene
        return bp_exon_row_meta_rep
    def AssignVar2Exon4(self, bp_exon_row_meta, VarFile, IntersectionWithPredicted = True, ProSib="Pro"):
        VarDF = pd.read_csv(VarFile, delimiter="\t")
        # Load annotations to gene
        entrez_symbol_map = get_gene_entrez_symbol_map()
        wigler_predicted_lgd_genes = set([entrez_symbol_map[x.strip()] for x in file(wigler_predicted_lgd)])
        channel_psd_genes = set([x.strip() for x in file(andy_psd_channel)])
        chromatin_genes = set([x.strip() for x in file(andy_snf_bromo)])
        cyto_sign = set([x.strip() for x in file(andy_cytp_sign)])
        tf_rbp = set([x.strip() for x in file(andy_znf)])
        
        Cates = dict(zip(["Channel&PSD", "Chromatin", "Cytp&Sig", "TF&RBP"],[channel_psd_genes,chromatin_genes,cyto_sign,tf_rbp]))
        res = {} # k:gene v:list of variants in that gene
        for i, row in VarDF.iterrows():
            if IntersectionWithPredicted:
                if row["Gene"] not in wigler_predicted_lgd_genes:
                    continue
            gene,var = row["Gene"], str(row["#Chr"])+":"+str(row["Pos"])
            if gene not in res:
                res[gene] = [var]
            else:
                res[gene].append(var)
            # Get IQ Info
        bp_exon_row_meta_rep = bp_exon_row_meta.copy(deep=True)
        bp_exon_row_meta_rep["Func"] = ""
        bp_exon_row_meta_rep["Vars"] = ""
        bp_exon_row_meta_rep["GeneHited"] = "F"
        bp_exon_row_meta_rep["Last"] = "F"
        LAST_GENE = None
        for i, row in bp_exon_row_meta_rep.iterrows():
            sys.stdout.write("\r{}".format(i))
            gene = row["gene_symbol"]
            if gene != LAST_GENE and gene != None:
                bp_exon_row_meta_rep.at[i-1, "Last"] = "T" 
            LAST_GENE = gene
            if gene not in res:
                continue
            bp_exon_row_meta_rep.at[i, "GeneHited"] = "T"
            start, end = int(row["start"]) - 2 , int(row["end"]) + 2
            gene = row["gene_symbol"]
            # Assign Variant to Exon
            for var in res[gene]:
                pos = int(var.split(":")[1])
                #print pos
                if pos >= start and pos <= end: #Var In This Exon
                    bp_exon_row_meta_rep.at[i, "Vars"] = bp_exon_row_meta_rep.get_value(i, "Vars") + ";" + var if bp_exon_row_meta_rep.get_value(i, "Vars") != "" else var
                    continue
            # Assign Other Info
            bp_exon_row_meta_rep.at[i, "Func"] = self.AssignFunc(gene, Cates)
        bp_exon_row_meta_rep.at[i, "Last"] = "T" #Last Exon in DF is last exon of that gene
        return bp_exon_row_meta_rep
    def AssignVar2Exon5(self, bp_exon_row_meta, VarFile, IntersectionWithPredicted = True, ProSib="Pro"):
        VarDF = VarFile 
        # Load annotations to gene
        entrez_symbol_map = get_gene_entrez_symbol_map()
        wigler_predicted_lgd_genes = set([entrez_symbol_map[x.strip()] for x in file(wigler_predicted_lgd)])
        channel_psd_genes = set([x.strip() for x in file(andy_psd_channel)])
        chromatin_genes = set([x.strip() for x in file(andy_snf_bromo)])
        cyto_sign = set([x.strip() for x in file(andy_cytp_sign)])
        tf_rbp = set([x.strip() for x in file(andy_znf)])
        
        Cates = dict(zip(["Channel&PSD", "Chromatin", "Cytp&Sig", "TF&RBP"],[channel_psd_genes,chromatin_genes,cyto_sign,tf_rbp]))
        res = {} # k:gene v:list of variants in that gene
        for i, row in VarDF.iterrows():
            if IntersectionWithPredicted:
                if row["Gene"] not in wigler_predicted_lgd_genes:
                    continue
            gene,var = row["Gene"], str(row["#Chr"])+":"+str(row["Pos"])
            if gene not in res:
                res[gene] = [var]
            else:
                res[gene].append(var)
            # Get IQ Info
        bp_exon_row_meta_rep = bp_exon_row_meta.copy(deep=True)
        bp_exon_row_meta_rep["Func"] = ""
        bp_exon_row_meta_rep["Vars"] = ""
        bp_exon_row_meta_rep["GeneHited"] = "F"
        bp_exon_row_meta_rep["Last"] = "F"
        LAST_GENE = None
        for i, row in bp_exon_row_meta_rep.iterrows():
            sys.stdout.write("\r{}".format(i))
            gene = row["gene_symbol"]
            if gene != LAST_GENE and gene != None:
                bp_exon_row_meta_rep.at[i-1, "Last"] = "T" 
            LAST_GENE = gene
            if gene not in res:
                continue
            bp_exon_row_meta_rep.at[i, "GeneHited"] = "T"
            start, end = int(row["start"]) - 2 , int(row["end"]) + 2
            gene = row["gene_symbol"]
            # Assign Variant to Exon
            for var in res[gene]:
                pos = int(var.split(":")[1])
                #print pos
                if pos >= start and pos <= end: #Var In This Exon
                    bp_exon_row_meta_rep.at[i, "Vars"] = bp_exon_row_meta_rep.get_value(i, "Vars") + ";" + var if bp_exon_row_meta_rep.get_value(i, "Vars") != "" else var
                    continue
            # Assign Other Info
            bp_exon_row_meta_rep.at[i, "Func"] = self.AssignFunc(gene, Cates)
        bp_exon_row_meta_rep.at[i, "Last"] = "T" #Last Exon in DF is last exon of that gene
        return bp_exon_row_meta_rep
    def AssignVar2Exon6(self, bp_exon_row_meta, VarFile, IntersectionWithPredicted = True, ProSib="Pro"):
        VarDF = pd.read_excel(VarFile)
        # Load annotations to gene
        entrez_symbol_map = get_gene_entrez_symbol_map()
        wigler_predicted_lgd_genes = set([entrez_symbol_map[x.strip()] for x in file(wigler_predicted_lgd)])
        fam_info = pd.read_excel(wigler_fam_info)

        res = {} # k:gene v:list of variants in that gene
        var_genders = {}
        var_IQ = {}
        for i, row in VarDF.iterrows():
            if IntersectionWithPredicted:
                if row["effectGene"] not in wigler_predicted_lgd_genes:
                    continue
            # Get Variant Info
            gene, fam, var = row["effectGene"], rwo["familyId"], row["vcfVariant"]
            key = fam + "-" + var 
            if gene not in res:
                res[gene] = [row["location"]]
            else:
                res[gene].append(row["location"])
            # Get Gender Info
            if var not in var_genders:
                if ProSib == 'Pro':
                    var_genders[var] = row["inChild"][:2]
                else:
                    var_genders[var] = row["inChild"][-2:]
                #var_genders[var] = [row["inChild"][:2]]
            else:
                if ProSib == 'Pro':
                    var_genders[var] = var_genders[var] + ";" + row["inChild"][:2]
                else:
                    var_genders[var] = var_genders[var] + ";" + row["inChild"][-2:]
            # Get IQ Info
            if var not in var_IQ:
                FamID = row["familyId"]
                var_IQ[var] = (famID2VIQ[FamID],famID2NVIQ[FamID])
        bp_exon_row_meta_rep = bp_exon_row_meta.copy(deep=True)
        bp_exon_row_meta_rep["Vars"] = ""
        bp_exon_row_meta_rep["GeneHited"] = "F"
        for i, row in bp_exon_row_meta_rep.iterrows():
            sys.stdout.write("\r{}".format(i))
            gene = row["gene_symbol"]
            if gene not in res:
                continue
            bp_exon_row_meta_rep.at[i, "GeneHited"] = "T"
            start, end = int(row["start"]) - 2 , int(row["end"]) + 2
            gene = row["gene_symbol"]
            # Assign Variant to Exon
            for var in res[gene]:
                pos = int(var.split(":")[1])
                #print pos
                if pos >= start and pos <= end: #Var In This Exon
                    bp_exon_row_meta_rep.at[i, "Vars"] = bp_exon_row_meta_rep.get_value(i, "Vars") + ";" + var if bp_exon_row_meta_rep.get_value(i, "Vars") != "" else var
                    continue
            # Assign Other Info
        bp_exon_row_meta_rep.at[i, "Last"] = "T" #Last Exon in DF is last exon of that gene
        return bp_exon_row_meta_rep

    def LookMutationTargetedExon(self, Gene, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, GeneDat=None, smooth=True, drop_low_exp=True, fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        GeneExonExp = ExonExp[ExonExp[0].isin(list(Exons["row_num"]))]
        exon_ids = list(Exons["row_num"])
        plt.close('all')
        fig = plt.figure(dpi=800)
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : fontsize}
        mpl.rc('font', **font)
        from mpl_toolkits.axes_grid1 import Grid
        res = {}
        for exon_id in exon_ids: # Plot each exon
            stages = {}
            for j, structure_acronym in enumerate(structure_acronyms): # sum over structures/regions
                region_exon_meta = bp_exon_col_meta[bp_exon_col_meta["structure_acronym"]==structure_acronym]
                for index, row in region_exon_meta.iterrows(): 
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    stages[Period].add_exp(GeneExonExp.get_value(exon_id-1, row["column_num"]))
            seq = DisplayStageExp(stages)[0]
            if drop_low_exp:
                if seq.count(0) >= 4:
                    res[exon_id] = None
                    continue
            if smooth:
                res[exon_id] = smooth_func(seq)
            else:
                res[exon_id] = seq
        fig, ax = plt.subplots(dpi=200)
        plt.title("Gene:{} over Region:{}".format(Gene, ",".join(structure_acronyms)))
        for exon_id in exon_ids: 
            if res[exon_id] == None:
                continue
            ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]], label=exon_id)
        if GeneDat != None:# Plot Gene exp over stages 
            GeneExp, GeneRow, GeneCol = GeneDat
            Genes = GeneRow[GeneRow["gene_symbol"]==Gene]
            Gene = GeneExp[GeneExp[0].isin(list(Genes["row_num"]))]
            gene_ids = list(Genes["row_num"])
            for gene_id in gene_ids:
                for j, structure_acronym in enumerate(structure_acronyms): # sum over structures/regions
                    region_meta = GeneCol[GeneCol["structure_acronym"]==structure_acronym]
                    for index, row in region_meta.iterrows():
                        Period = row["Period"]
                        if Period not in stages:
                            stages[Period] = DevStageExp(Period)
                        stages[Period].add_exp(GeneExp.get_value(gene_id-1, row["column_num"]))
                seq, Nsample = DisplayStageExp(stages)
                print (seq)
                if smooth:
                    res[gene_id] = smooth_func(seq)
                else:
                    res[gene_id] = seq
            #add_layout(LinearAxis(y_range_name="GeneRPKM"), 'right')
            ax2 = ax.twinx()
            for gene_id in gene_ids:
                print (res[gene_id])
                ax2.plot(range(2,14), [math.log(x,2) if x !=0 else 0 for x in res[gene_id]], color="black") 
                #ax2.plot(range(2,14), res[gene_id], color="black") 
            ax2.set_ylabel("GeneRPKM")
        ax.legend()
        ax.grid(True)
        ax.axvline(x=7.5)
        #Stages = Stages + [0,0,0,0,0]
        plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        plt.show()
    def LookMutationTargetedGene(self, Gene, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6):
        Exons = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==Gene]
        #print Exons
        GeneExonExp = ExonExp[ExonExp[0].isin(list(Exons["row_num"]))]
        #ID_start, ID_end = list(Exons["row_num"])[0], list(Exons["row_num"])[-1]
        exon_ids = list(Exons["row_num"])
        #print exon_ids
        plt.close('all')
        fig = plt.figure(dpi=800)
        font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : fontsize}
        mpl.rc('font', **font)
        from mpl_toolkits.axes_grid1 import Grid
        #grid = Grid(fig, rect=111, nrows_ncols=(XX,YY), axes_pad=0.25, label_mode='L',)
        res = {}
        for exon_id in exon_ids:
            stages = {}
            for j, structure_acronym in enumerate(structure_acronyms):
                region_exon_meta = bp_exon_col_meta[bp_exon_col_meta["structure_acronym"]==structure_acronym]
                for index, row in region_exon_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    stages[Period].add_exp(GeneExonExp.get_value(exon_id-1, row["column_num"]))
            #res[exon_id]
            seq = DisplayStageExp(stages)[0]
            if drop_low_exp:
                if seq.count(0) >= 4:
                    res[exon_id] = None
                    continue
            if smooth:
                res[exon_id] = smooth_func(seq)
            else:
                res[exon_id] = seq

        fig, ax = plt.subplots(dpi=200)
        plt.title("Gene:{} over Region:{}".format(Gene, ",".join(structure_acronyms)))
        for exon_id in exon_ids: 
            if res[exon_id] == None:
                continue
            #if np.array([math.log(x, 2) if x!=0 else 0 for x in res[exon_id]]).
            ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in res[exon_id]])
            #ax.plot(range(2,14),res[exon_id])
            #print len(res[exon_id]), res[exon_id]
            #ax.plot(range(2,14),res[exon_id])
        ax.grid(True)
        ax.axvline(x=7.5)
        #Stages = Stages + [0,0,0,0,0]
        plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        plt.show()
    def LoadingDat2SeqWithLenCrossRecordCrossRegion(self, record_ids, regions, row_meta, col_meta, Matrix, smooth=True, drop_low_exp=True):
        lengths = []
        stages = {}
        for i, _id in enumerate(record_ids):
            sys.stdout.write("\rLoading records:{}".format(i))
            lengths.append(row_meta.get_value(int(_id)-1, "exon length"))
            for j, region in enumerate(regions):
                region_col_meta = col_meta[col_meta["structure_acronym"]==region]
                for index, row in region_col_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    stages[Period].add_exp(Matrix.get_value(_id-1, row["column_num"]))
        sys.stdout.write("\n")
        seq = self.DisplayStageExp(stages)[0]
        if smooth:
            seq = self.smooth_func(seq)
        return seq, lengths
    def LoadingNormDat2SeqCrossRecordCrossRegion(self, record_ids, regions, row_meta, col_meta, Matrix, smooth=True, drop_low_exp=True):
        res = {}
        ALL_RPKM_X_L = [0]*13 # by Timepoint
        ALL_L = 0
        Numrecords = len(record_ids)
        for i, _id in enumerate(record_ids):
            sys.stdout.write("\rLoading records:{}".format(i))
            #lengths.append(row_meta.get_value(int(_id)-1, "exon length"))
            exonL = int(row_meta.get_value(int(_id)-1, "exon length"))
            stages = {} #Exom RPKM at each stage
            for j, region in enumerate(regions):
                region_col_meta = col_meta[col_meta["structure_acronym"]==region]
                for index, row in region_col_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    stages[Period].add_exp(Matrix.get_value(_id-1, row["column_num"]))
            seq = self.DisplayStageExp(stages)[0]
            if drop_low_exp:
                if seq.count(0) >= 4:
                    print (_id)
                    Numrecords -= 1
                    continue
            #print seq
            res[_id] = self.smooth_func(seq) if smooth else seq
            ALL_RPKM_X_L = [x+(y*exonL) for x,y in zip(ALL_RPKM_X_L, res[_id])]
            ALL_L += exonL
            #print ALL_RPKM_X_L, ALL_L
        records_mean = []
        records_mean_norm = []
        denominators = []
        for i, stage in enumerate(self.Stages):
            tmp1 = 0
            tmp2 = 0
            denominator = ALL_RPKM_X_L[i] / ALL_L
            denominators.append(denominator)
            expsi = []
            for j, _id in enumerate(record_ids):
                if _id in res:
                    #pprint stage, res[_id][i]
                    tmp1 += res[_id][i]
                    res[_id][i] = res[_id][i] / denominator
                    tmp2 += res[_id][i]
                    expsi.append(res[_id][i])
            print (np.var(expsi),)
            records_mean.append(tmp1/Numrecords)
            records_mean_norm.append(tmp2/Numrecords)
            print()
        print (records_mean, records_mean_norm, denominators)
        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(xrange(2,14), records_mean, label='avg')
        ax.plot(xrange(2,14), denominators, label='norm term')
        ax2 = ax.twinx()
        ax2.plot(xrange(2,14), records_mean_norm, label='avg_nromed')
        ax.legend()
        ax2.legend()
        plt.show()
        sys.stdout.write("\n")
        print (records_mean )
        return records_mean 
    def LoadingDat2SeqCrossRecordCrossRegion(self, record_ids, regions, row_meta, col_meta, Matrix, smooth=True, drop_low_exp=True, space = 'norm', shutup=True):
        stages = {}
        for i, _id in enumerate(record_ids):
            if not shutup:
                sys.stdout.write("\rLoading records:{}".format(i))
            for j, region in enumerate(regions):
                region_col_meta = col_meta[col_meta["structure_acronym"]==region]
                for index, row in region_col_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = DevStageExp(Period)
                    exp = Matrix.get_value(_id-1, row["column_num"])
                    if space == 'log':
                        exp = math.log(exp+1, 2)
                    stages[Period].add_exp(exp)
        if not shutup:
            sys.stdout.write("\n")
        seq, Nsample, error_var, median = self.DisplayStageExp(stages)
        if smooth:
            seq = self.smooth_func(seq)
        return seq, error_var, median 
    def LookALLMutationTargetedExon2(self, exon_ids, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6, rd=False):
        plt.close('all')
        fig = plt.figure(dpi=800)
        stages = {}
        ExonLengths = []
        seq, lengths = LoadingDat2SeqCrossRecordCrossRegion(exon_ids, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True) # Loading data needed
        fig, ax = plt.subplots(dpi=200)
        plt.title("TargetedExons over Region:{}".format(",".join(structure_acronyms)))
        ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in seq], label="TargetedExons")
        ax.grid(True)
        ax.axvline(x=7.5)
        if rd != False:
            if rd == "gt1000":
                rd_exon_ids = list(bp_exon_row_meta[bp_exon_row_meta["exon length"]>=1000].sample(len(exon_ids)*10)["row_num"])
            if rd == "lt1000":
                rd_exon_ids = list(bp_exon_row_meta[bp_exon_row_meta["exon length"]<1000].sample(len(exon_ids)*10)["row_num"])
            rd_seq, rd_length = LoadingDat2SeqCrossRecordCrossRegion(rd_exon_ids,structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True)
            ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in rd_seq], label="RandomExons")
        plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        #ax.legend()
        plt.show()
        return ExonLengths
    def LookALLMutationTargetedExon(self, exon_ids_sets, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6, rd=False, method = "mean", space = 'log'):
        plt.close('all')
        fig = plt.figure(dpi=800)
        stages = {}
        fig, ax = plt.subplots(dpi=200)
        plt.title("Exons Exp over Region:{}".format(",".join(structure_acronyms)))
        seq1, seq2 = None, None

        for SetName, (color, ExonIDs) in exon_ids_sets.items():
            print (SetName)
            ExonIDs = [int(x) for x in ExonIDs]
            seq, error_var, median = self.LoadingDat2SeqCrossRecordCrossRegion(ExonIDs, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=False, space = space, shutup=False) # Loading data needed
            #seq = [math.log(x, 10) for x in seq]
            if "Other" in SetName:
                if method == "mean":
                    ax.errorbar(range(2,14), seq, yerr = error_var, linestyle = '--', label=SetName, color=color)
                    seq2 = seq
                elif method == "median":
                    ax.plot(range(2,14), median, '--', label=SetName, color=color)
                    seq2 = median
            else:
                if method == "mean":
                    ax.errorbar(range(2,14), seq, yerr = error_var, label=SetName, color=color)
                    seq1 = seq
                elif method == "median":
                    ax.plot(range(2,14),median, label=SetName, color=color)
                    seq1 = median
        ax.grid(True)
        ax.axvline(x=7.5)
        plt.xticks(np.arange(2,14), self.Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        ax.legend()
        plt.show()
        pre, post = self.Bias(seq1, seq2)
        print (pre, post, pre-post)
    # with Nromlization within exon sets.
    def LookALLMutationTargetedExon(self, exon_ids_pairs, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6, rd=False, method = "mean", space = 'log'):
        plt.close('all')
        fig = plt.figure(dpi=800)
        stages = {}
        fig, ax = plt.subplots(dpi=200)
        plt.title("Exons Exp over Region:{}".format(",".join(structure_acronyms)))
        for (SetName, color, TExonIDs, UExonIDs) in exon_ids_pairs:
            seq1, seq2 = None, None
            TExonIDs = [int(x) for x in TExonIDs]
            UExonIDs = [int(x) for x in UExonIDs]
            Tseq, Terror_var, Tmedian = self.LoadingDat2SeqCrossRecordCrossRegion(TExonIDs, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=False, space = space, shutup=False) # Loading data needed
            Useq, Uerror_var, Umedian = self.LoadingDat2SeqCrossRecordCrossRegion(UExonIDs, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=False, space = space, shutup=False) # Loading data needed
            Tseq = [2 ** x for x in Tseq]
            Useq = [2 ** x for x in Useq]
            #Terror_var = self.converterror(Tseq, Terror_var)
            #Uerror_var = self.converterror(Useq, Uerror_var)
            #seq = [math.log(x, 10) for x in seq]
            if method == "mean":
                ax.errorbar(range(2,14), Tseq, yerr = Terror_var, label=SetName+"_Targeted", color=color)
                ax.errorbar(range(2,14), Useq, yerr = Uerror_var, linestyle = '--', label=SetName+"_Untargeted", color=color)
                seq1 = Tseq
                seq2 = Useq
            elif method == "median":
                ax.plot(range(2,14), Tmedian, '--', label=SetName, color=color)
                ax.plot(range(2,14), Umedian, '--', label=SetName, color=color)
                seq1 = Tmedian
                seq2 = Umedian
            pre, post = self.Bias(seq1, seq2)
            print (SetName, pre, post, pre-post)
        ax.grid(True)
        ax.axvline(x=7.5)
        plt.xticks(np.arange(2,14), self.Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        ax.legend()
        plt.show()
    def converterror(self, means, stderrs):
        upper, lower = [], []
        for mean, error in zip(means, stderrs):
            upper.append( (2**(mean+error)) - (2**(mean)) )
            lower.append( (2**(mean)) - (2**(mean-error)) )
        #print upper, lower
        return upper, lower
    # with Nromlization within exon sets.
    def ReadExonExpValuesEachTimePointAvgAcrossRegion(self, exon_ids, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True):
        res = {}
        for ExonID in exon_ids:
            seq = self.LoadingDat2SeqCrossRecordCrossRegion([ExonID], structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True) # Loading data needed
            res[ExonID] = seq
        return res
    def LookALLMutationTargetedExon3(self, exon_ids_sets, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True, drop_low_exp=True, fontsize=6, rd=False):
        plt.close('all')
        fig = plt.figure(dpi=800)
        stages = {}
        fig, ax = plt.subplots(dpi=200)
        plt.title("Exons Exp over Region:{}".format(",".join(structure_acronyms)))
        for SetName, (color, ExonIDs) in exon_ids_sets.items():
            print (SetName)
            ExonIDs = [int(x) for x in ExonIDs]
            seq = self.LoadingNormDat2SeqCrossRecordCrossRegion(ExonIDs, structure_acronyms, bp_exon_row_meta, bp_exon_col_meta, ExonExp, smooth=True) # Loading data needed
            seq = [math.log(x,10) for x in seq]
            if "Other" in SetName:
                ax.plot(range(2,14), seq, '--', label=SetName, color=color)
                #ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in seq], '--', label=SetName, color=color)
            else:
                ax.plot(range(2,14), seq, label=SetName, color=color)
                #ax.plot(range(2,14),[math.log(x, 2) if x!=0 else 0 for x in seq], label=SetName, color=color)
        ax.grid(True)
        ax.axvline(x=7.5)
        plt.xticks(np.arange(2,14), self.Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        ax.legend()
        plt.show()
    def LoadSummarizedExonExp(self, record_ids, bp_exon_row_meta, ExonExp, drop_low_exp = True):
        stages = {}
        drops = []
        for stage in self.Stages:
            stages[stage] = []
        N = 0
        Ndrop = 0
        for i, _id in enumerate(record_ids):
            dat = ExonExp.iloc[_id-1, 1:13].values
            if drop_low_exp and dat.mean() < 1 :
                Ndrop += 1
                drops.append(_id)
                continue
            for j, stage in enumerate(self.Stages):
                #stages[stage].append(dat[j])
                stages[stage].append(math.log(dat[j]+1, 2))
            N += 1
        log2seq = []
        stderrs = []
        medians = []
        for i, stage in enumerate(self.Stages):
            log2seq.append(np.mean(stages[stage]))
            stderrs.append(math.sqrt( np.var(stages[stage])/N ))
            medians.append(np.median(stages[stage]))
        print (Ndrop)
        return log2seq, stderrs, medians, drops
    def Bias(self, seq1, seq2):
        diff = []
        for i, stage in enumerate(self.Stages):
            diff.append(seq1[i]-seq2[i])
        pre, post = sum(diff[0:6]), sum(diff[6:12])
        return pre, post
    def LookALLMutationTargetedExon4(self, title, exon_ids_sets, bp_exon_row_meta, ExonExp, method = "mean", smooth=True, drop_low_exp=True, fontsize=6):
        plt.close('all')
        fig, ax = plt.subplots(dpi=120)
        plt.title("Exons Exp {}".format(title))
        seq1, seq2 = None, None
        ALL_drop = []
        for SetName, (color, ExonIDs) in exon_ids_sets.items():
            print (SetName)
            ExonIDs = [int(x) for x in ExonIDs]
            log2seq, stderrs, medians, drops = self.LoadSummarizedExonExp(ExonIDs, bp_exon_row_meta, ExonExp, drop_low_exp=drop_low_exp) # Loading data needed
            ALL_drop.extend(drops)
            if "Other" in SetName:
                if method == "median":
                    ax.plot(range(2,14), medians, '--', label=SetName, color=color)
                    seq2 = medians
                else:
                    ax.errorbar(range(2,14), log2seq, yerr = stderrs, linestyle = '--', label=SetName, color=color)
                    seq2 = log2seq
            else:
                if method == "median":
                    ax.plot(range(2,14), medians, label=SetName, color=color)
                    seq1 = medians
                else:
                    ax.errorbar(range(2,14), log2seq , yerr = stderrs, label=SetName, color=color)
                    seq1 = log2seq
        ax.grid(True)
        ax.axvline(x=7.5)
        plt.xticks(np.arange(2,14), self.Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        ax.legend()
        plt.show()
        pre, post = self.Bias(seq1, seq2)
        print (pre, post, pre-post)
        return ALL_drop

    def LookALLMutationTargetedGenes(self, Gene_id_sets, structure_acronyms, GeneDat, smooth=True, drop_low_exp=True, fontsize=6, ylim=None):
        plt.close('all')
        GeneExp, GeneRow, GeneCol = GeneDat
        plt.close('all')
        stages = {}
        fig, ax = plt.subplots(dpi=200)
        plt.title("TargetedExons over Region:{}".format(",".join(structure_acronyms)))
        for SetName, (color, GeneIDs) in Gene_id_sets.items():
            print (SetName)
            GeneIDs = [int(x) for x in GeneIDs]
            seq = self.LoadingDat2SeqCrossRecordCrossRegion(GeneIDs, structure_acronyms, GeneCol, GeneCol, GeneExp, smooth=True) # Loading data needed
            #ax.plot(range(1,16),[math.log(x, 2) if x!=0 else 0 for x in seq])
            ax.plot(range(2,14), seq, label=SetName, color=color)
        ax.grid(True)
        ax.axvline(x=7.5)
        plt.legend()
        #plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        if ylim != None:
            plt.ylim(ylim)
        plt.show()
    def AssignVar2Gene(self, bp_gene_row_meta, VarFile):
        reader = csv.reader(open(VarFile, 'rb'))
        header = reader.next()
        res = {} # k:gene v:list of variants in that gene
        for row in reader:
            tmp = dict(zip(header, row))
            if int(tmp["GeneCount"]) < 2:
                continue
            if tmp["Gene"] not in res:
                res[tmp["Gene"]] = [tmp]
            else:
                res[tmp["Gene"]].append(tmp)
        bp_gene_row_meta["Vars"] = ""
        for i, row in bp_gene_row_meta.iterrows():
            sys.stdout.write("\r{}".format(i))
            if row["gene_symbol"] not in res:
                continue
            else:
                for var in res[row["gene_symbol"]]:
                    bp_gene_row_meta.at[i, "Vars"] = bp_gene_row_meta.get_value(i, "Vars") + ";" + var["cDnaVariant"]
    def AssignVar2Gene2(self,bp_gene_row_meta, VarFile):
        df = pd.read_excel(VarFile, sheetname="Proband SNVs")
        

        res = {} # k:gene v:list of variants in that gene
        for row in reader:
            tmp = dict(zip(header, row))
            if tmp["Gene"] not in res:
                res[tmp["Gene"]] = [tmp]
            else:
                res[tmp["Gene"]].append(tmp)
        bp_gene_row_meta["Vars"] = ""
        for i, row in bp_gene_row_meta.iterrows():
            sys.stdout.write("\r{}".format(i))
            if row["gene_symbol"] not in res:
                continue
            else:
                for var in res[row["gene_symbol"]]:
                    bp_gene_row_meta.at[i, "Vars"] = bp_gene_row_meta.get_value(i, "Vars") + ";" + var["cDnaVariant"]
    def format_func2(self, value, tick_number):
        try:
            return Stages[int(value)]
        except:
            #print value
            return 0
    def DisplayGeneExpViolin(self, Gene, GeneDat, structure_acronyms):
        GeneExp, GeneRow, GeneCol = GeneDat
        SelectedGenes = GeneRow[GeneRow["gene_symbol"]==Gene]
        #SelectGeneExp = GeneExp[GeneExp[0].isin(list(SelectedGenes["row_num"]))]
        gene_ids = list(SelectedGenes["row_num"])
        res = {}
        for gene_id in gene_ids:
            res[gene_id] = {}
            stages = {}
            for j, structure_acronym in enumerate(structure_acronyms): # sum over structures/regions
                region_meta = GeneCol[GeneCol["structure_acronym"]==structure_acronym]
                for index, row in region_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = []
                    stages[Period].append(GeneExp.get_value(gene_id-1, row["column_num"]))
            res[gene_id] = stages
        for gene_id in gene_ids:
            Xs, dat = [], []
            for stage in Stages:
                Xs.append(stage)
                dat.append(res[gene_id][stage])
            #ax = sns.boxplot(x=Xs, data=dat)
            ax = sns.violinplot(data=dat)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func2))
            plt.show()
    def DisplayGeneSumExpViolin(self, Genes, GeneDat, structure_acronyms, ylim=None):
        GeneExp, GeneRow, GeneCol = GeneDat
        SelectedGenes = GeneRow[GeneRow["gene_symbol"].isin(Genes)]
        #SelectGeneExp = GeneExp[GeneExp[0].isin(list(SelectedGenes["row_num"]))]
        gene_ids = list(SelectedGenes["row_num"])
        stages = {}
        for i, gene_id in enumerate(gene_ids):
            sys.stdout.write("\rLoading Genes {}".format(i))
            for j, structure_acronym in enumerate(structure_acronyms): # sum over structures/regions
                region_meta = GeneCol[GeneCol["structure_acronym"]==structure_acronym]
                for index, row in region_meta.iterrows():
                    Period = row["Period"]
                    if Period not in stages:
                        stages[Period] = []
                    stages[Period].append(GeneExp.get_value(gene_id-1, row["column_num"]))
        dat = []
        for stage in Stages:
            #print stage
            try: 
                #dat.append([ math.log(2,x) for x in stages[stage]])
                dat.append(stages[stage])
            except:
                dat.append([0])
            #dat.append(np.mean(stages[stage]))
        plt.close('all')
        ax = sns.violinplot(data=dat)
        #ax = sns.boxplot(data=dat)
        #ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func2))
        #plt.plot(xrange(12), dat)
        if ylim != None:
            plt.ylim(ylim)
        plt.show()

    def LoadGeneSetData2Fil(self, FilName, Genes, Regions, Exon_row_meta, Exon_col_meta, Matrix):
        #Regions_indice
        writer = csv.writer(open(FilName, 'wt'))
        print ("Total Num of Genes:", len(Genes))
        for i, gene in enumerate(Genes):
            ExonIDs = Exon_row_meta[Exon_row_meta["gene_symbol"]==gene]["row_num"].values
            for ExonID in ExonIDs:
                ExonID = int(ExonID)
                for i, stage in enumerate(self.Stages):
                    tmp = [gene, ExonID, stage]
                    cols = Exon_col_meta[(Exon_col_meta["Period"]==stage) & (Exon_col_meta["structure_acronym"].isin(Regions))]["column_num"].values
                    for col in cols:
                        exp = Matrix.get_value(ExonID-1, col)
                        exp = round(math.log(exp+1, 2), 6)
                        tmp.append(exp)
                    writer.writerow(tmp)
            #if i % 10 == 0:
            #sys.stdout.write("\rLoad {} genes".format(i))
    
    # Similar to previous func, gene stage instead of gene exon stage
    def LoadGeneSetData2Fil2(self, FilName, Genes, Regions, row_meta, col_meta, Matrix):
        #Regions_indice
        writer = csv.writer(open(FilName, 'wt'))
        print ("Total Num of Genes:", len(Genes))
        for i, gene in enumerate(Genes):
            #ExonIDs = Exon_row_meta[Exon_row_meta["gene_symbol"]==gene]["row_num"].values
            GeneID = row_meta[row_meta["gene_symbol"]==gene]["row_num"].values[0]
            #for ExonID in ExonIDs:
            #    ExonID = int(ExonID)
            for i, stage in enumerate(self.Stages):
                tmp = [gene, stage]
                cols = col_meta[(col_meta["Period"]==stage) & (col_meta["structure_acronym"].isin(Regions))]["column_num"].values
                for col in cols:
                    exp = Matrix.get_value(GeneID-1, col)
                    exp = round(math.log(exp+1, 2), 6)
                    tmp.append(exp)
                writer.writerow(tmp)
            #if i % 10 == 0:
            #sys.stdout.write("\rLoad {} genes".format(i))

    def LoadGeneSetDataFromFil(self, FilName):
        res = {}
        fin = open(FilName, 'rt')
        for l in fin:
            llist = l.strip().split(",")
            Gene, Exon, time = llist[:3]
            Exon = int(Exon)
            #exps = map(float, llist[3:])
            exps = [float(x) for x in llist[3:]]
            if Gene not in res:
                res[Gene] = {}
            if Exon not in res[Gene]:
                res[Gene][Exon] = {}
            if time not in res[Gene][Exon]:
                res[Gene][Exon][time] = None
            res[Gene][Exon][time] = exps
        return res

    def LoadGeneSetDataFromFil2(self, FilName):
        res = {}
        fin = open(FilName, 'rt')
        for l in fin:
            llist = l.strip().split(",")
            Gene, time = llist[:2]
            #exps = map(float, llist[2:])
            exps = [float(x) for x in llist[2:]]
            if Gene not in res:
                res[Gene] = {}
            if time not in res[Gene]:
                res[Gene][time] = None
            res[Gene][time] = exps
        return res

class GeneExonSet(BrainSpan):
    def __init__(self, expdict, Name="geneset", Color='black'):
        self.GeneSetName = Name
        self.GeneSetColor = Color
        self.Genes = {} # a list of genes involved
        self.expdict = expdict
        self.genes = []
        self.TargetedExon = []
        self.UntargetedExon = []
    def addGene(self, Genesymbol, TargetedExon, UntargetedExon, TargetedExonLength, UntargetedExonLength):
        self.Genes[Genesymbol] = TargetedGene(Genesymbol, TargetedExon, UntargetedExon, TargetedExonLength, UntargetedExonLength)
        self.genes.append(Genesymbol)
        self.TargetedExon.extend(TargetedExon)
        self.UntargetedExon.extend(UntargetedExon)
        #self.Genes[Genesymbol].addExp(ExpDict)

    def Reduce(self, logscale = False):
        self.Tseq, self.Terr, self.Useq, self.Uerr, self.All, self.Allerr= [], [], [], [], [], []
        for stage in Stages:
            tmp1 = []
            tmp2 = []
            tmp3 = []
            for gene in self.Genes:
                if gene not in self.expdict or gene not in self.Genes:
                    continue 
                for ExonID in self.Genes[gene].TargetedExons:
                    tmp1.extend(self.expdict[gene][ExonID][stage])
                    tmp3.extend(self.expdict[gene][ExonID][stage])
                for ExonID in self.Genes[gene].UntargetedExons:
                    tmp2.extend(self.expdict[gene][ExonID][stage])
                    tmp3.extend(self.expdict[gene][ExonID][stage])
            self.Tseq.append(np.mean(tmp1))
            self.Terr.append( math.sqrt( np.var(tmp1) / len(tmp1) ) )
            self.Useq.append(np.mean(tmp2))
            self.Uerr.append( math.sqrt( np.var(tmp2) / len(tmp2) ) )
            self.All.append(np.mean(tmp3))
            self.Allerr.append( math.sqrt( np.var(tmp3) / len(tmp3) ) )
        if logscale == True:
            Tseq, Useq, Terr, Uerr, Allseq, Allerr = self.Tseq, self.Useq, self.Terr, self.Uerr, self.All, self.Allerr
        else:
            Tseq, Useq, Terr, Uerr = [2 ** x for x in self.Tseq], [2 ** x for x in self.Useq], self.converterror(self.Tseq, self.Terr), self.converterror(self.Useq, self.Uerr)
            Allseq, Allerr = [2 ** x for x in self.All], self.converterror(self.All, self.Allerr)
        return (Tseq, Terr, Useq, Uerr, Allseq, Allerr)
    def Reduce2(self):
        self.Tseq, self.Terr, self.Useq, self.Uerr = [], [], [], []
        for stage in Stages:
            tmp1 = []
            tmp2 = []
            for gene in self.Genes:
                for ExonID in self.Genes[gene].TargetedExons:
                    tmp1.extend(self.expdict[gene][ExonID][stage])
                for ExonID in self.Genes[gene].UntargetedExons:
                    tmp2.extend(self.expdict[gene][ExonID][stage])
            self.Tseq.append(np.mean(tmp1))
            self.Terr.append( math.sqrt( np.var(tmp1) / len(tmp1) ) )
            self.Useq.append(np.mean(tmp2))
            self.Uerr.append( math.sqrt( np.var(tmp2) / len(tmp2) ) )

    def Plot(self, title="", dpi=200, color='black'):
        plt.close('all')
        fig, ax = plt.subplots(dpi=dpi)
        plt.title(title)
        Tseq, Useq, Terr, Uerr = [2 ** x for x in self.Tseq], [2 ** x for x in self.Useq], self.converterror(self.Tseq, self.Terr), self.converterror(self.Useq, self.Uerr)
        #Tseq, Useq, Terr, Uerr = self.Tseq, self.Useq, self.Terr, self.Uerr
        ax.errorbar(range(2,15), Tseq, yerr = Terr, label=title+"_Targeted", color=color)
        ax.errorbar(range(2,15), Useq, yerr = Uerr, linestyle = '--', label=title+"_Untargeted", color=color)
        pre, post = self.Bias(Tseq, Useq)
        print (pre, post, pre-post)
        ax.grid(True)
        ax.axvline(x=7.5, color="black")
        plt.xticks(np.arange(2,15), Stages, rotation=20)
        #ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        ax.legend()
        plt.show()
    def converterror(self, means, stderr):
        upper, lower = [], []
        for mean, error in zip(means, stderr):
            upper.append( (2**(mean+error)) - (2**(mean)) )
            lower.append( (2**(mean)) - (2**(mean-error)) )
        #print upper, lower
        return upper, lower
    def wilcoxonTest(self):
        self.genePreBias = []    
        self.genePostBias = []    
        for g, gene in self.Genes.items():
            pre, post = gene.reduce(self.expdict)
            self.genePreBias.append(pre)
            self.genePostBias.append(post)
        return self.genePreBias, self.genePostBias

    def Permute(self, plot=False):
        Tseq, Useq = [], []
        tmp1, tmp2 = {}, {}
        TExons, UExons = [],[]
        for Gene in self.genes:
            GeneTargetedExons, GeneUntargetedExons = self.Genes[Gene].Permute()
            #print Gene, len(GeneTargetedExons), len(GeneUntargetedExons)
            #print GeneTargetedExons, GeneUntargetedExons
            TExons.extend(GeneTargetedExons)
            UExons.extend(GeneUntargetedExons)
            for stage in Stages:
                if stage not in tmp1:
                    tmp1[stage] = []
                if stage not in tmp2:
                    tmp2[stage] = []
                for ExonID in GeneTargetedExons:
                    tmp1[stage].extend(self.expdict[Gene][ExonID][stage])
                for ExonID in GeneUntargetedExons:
                    tmp2[stage].extend(self.expdict[Gene][ExonID][stage])
        for stage in Stages:
            Tseq.append(np.mean(tmp1[stage]))
            Useq.append(np.mean(tmp2[stage]))
        Tseq, Useq = [2 ** x for x in Tseq], [2 ** x for x in Useq]
        #pre, post, bias = Bias2(Tseq, Useq)
        pre1, post1, T, pre2, post2, U = Bias2(Tseq, Useq)
        #print pre, post, bias
        bias = T/U
        if plot:
            fig, ax = plt.subplots(dpi=80)
            ax.plot(range(2,14), Tseq, color='red')
            ax.plot(range(2,14), Useq, color='blue')
            ax.grid(True)
            ax.axvline(x=7.5, color="grey", linestyle="--")
            plt.show()
        return bias, TExons, UExons
        
def Bias(seq1, seq2, method=4):
    print ('bias')
    if method == 4:
        print ('4')
        pre1 = np.mean(seq1[0:6])
        post1 = np.mean(seq1[6:13])
        T = pre1/post1
        pre2 = np.mean(seq2[0:6])
        post2 = np.mean(seq2[6:13])
        U = pre2/post2
        return pre1, post1, T, pre2, post2, U
    elif method == 3:
        print ('3')
        FC_pre = []
        for T, U in zip(seq1[:6], seq2[:6]):
            FC_pre.append(T/U)
        FC_post = []
        for T, U in zip(seq1[6:12], seq2[6:12]):
            FC_post.append(T/U)
        FC_pre, FC_post = np.mean(FC_pre), np.mean(FC_post)
        return FC_pre, FC_post, FC_pre / FC_post
    else:
        print ('else')
        diff = []
        for i, stage in enumerate(Stages):
            diff.append(seq1[i]-seq2[i])
        #pre, post = sum(diff[0:6])/6, sum(diff[6:12])/6
        pre, post = sum(diff[0:6]), sum(diff[6:12])
        #return pre, post, (pre-post)/post
        return pre, post, (pre-post)
def Bias2(seq1, seq2, method=4):
    pre1 = np.mean(seq1[0:6])
    post1 = np.mean(seq1[6:13])
    T = pre1/post1
    pre2 = np.mean(seq2[0:6])
    post2 = np.mean(seq2[6:13])
    U = pre2/post2
    return pre1, post1, T, pre2, post2, U

class TargetedGene:
    def __init__(self, genesymbol, TargetedExonID, UntargetedExonID, TargetedExonLength, UntargetedExonLength):
        self.genesymbol = genesymbol
        self.TargetedExons = TargetedExonID
        self.NTargeted = len(TargetedExonID)
        self.UntargetedExons = UntargetedExonID
        self.TargetedExonLength = TargetedExonLength
        self.UntargetedExonLength = UntargetedExonLength
        self.AllExons = self.TargetedExons + self.UntargetedExons
        self.AllLength = self.TargetedExonLength + self.UntargetedExonLength
        #self.AllLength = [math.log(x, 1000) for x in self.AllLength]
        #self.AllLength = [min(500, x) for x in self.AllLength]
        self.TotalLength = sum(self.AllLength) 
        self.Weights = [float(x)/self.TotalLength for x in self.AllLength]
        #self.Weights = [1 for x in self.AllLength]
        #print self.Weights
        self.cdf_vals = cdf(self.Weights)
    def Permute(self):
        res_tar, res_untar = set([]), set([])
        i = 0
        while i < self.NTargeted:
            x = random.random()
            idx = bisect.bisect(self.cdf_vals, x)
            TheOne = self.AllExons[idx]
            #tmp = self.AllExons[::]
            #random.shuffle(tmp)
            #TheOne = tmp[0]
            #if TheOne not in res_tar:
            res_tar.add(TheOne)
            i += 1 
            #else:
            #    continue
        res_untar = set(self.AllExons).difference(res_tar)
        return list(res_tar), list(res_untar)
    def reduce(self, expdict):
        res = []
        for stage in Stages:
            u = np.mean([expdict[self.genesymbol][x][stage] for x in self.TargetedExons])
            l = np.mean([expdict[self.genesymbol][x][stage] for x in self.UntargetedExons])
            #u = 2**u
            #l = 2**l
            res.append(u-l)
            #res.append(2**(u-l))
        #print res
        if np.mean(res[:6]) - np.mean(res[6:12]) < -10:
            print (self.genesymbol, res )
        return np.mean(res[:6]), np.mean(res[6:12])
    def plot(self, expdict):
        fig, ax = plt.subplots(dpi=120)
        plt.title(self.genesymbol)
        Tseq, Useq, Uerr = [], [], [] 
        for stage in Stages:
            u = np.mean([expdict[self.genesymbol][x][stage] for x in self.TargetedExons])
            l = np.mean([expdict[self.genesymbol][x][stage] for x in self.UntargetedExons])
            l_err = math.sqrt(np.var([expdict[self.genesymbol][x][stage] for x in self.UntargetedExons])/len([expdict[self.genesymbol][x][stage] for x in self.UntargetedExons]))
            Uerr.append(l_err)
            #u = 2**u
            #l = 2**l
            Tseq.append(u)
            Useq.append(l)
            #res.append(2**(u-l))
        ax.plot(range(2,14), Tseq, label="Targeted", color='red')
        ax.errorbar(range(2,14), Useq, yerr = Uerr, linestyle = '--', label="Untargeted", color='grey')
        pre, post, bias = Bias(Tseq, Useq)
        print (pre, post, bias)
        ax.grid(True)
        ax.axvline(x=7.5, color="grey", linestyle="--")
        plt.xticks(np.arange(2,14), Stages, rotation=20)
        ax.legend(loc='upper right')
        plt.xlabel("Dev Stages")
        plt.ylabel("Expression")
        plt.show()


class TargetedGeneWithData:
    def __init__(self, genesymbol, TargetedExonID, UntargetedExonID):
        self.genesymbol = genesymbol
        self.TargetedExons = TargetedExonID
        self.NTargeted = len(TargetedExonID)
        self.UntargetedExons = UntargetedExonID
        #self.AllExons = [x for (x,y) in self.TargetedSet] + [x for (x,y) in self.UntargetedSet]
        #self.TotalLength = sum(y for (x,y) in self.TargetedSet) + sum(y for (x,y) in self.UntargetedSet)
        #self.Probs = dict(zip(self.AllExons, [float(y/self.totalLength) for (x,y) in self.TargetedExonID + self.UntargetedExonID]))
        #self.cdf_vals = cdf(weights)

    def addExp(self, ExpDict):
        for i, exonid, exonlength in enumerate(self.TargetedSet):
            self.Targetedset[i] = (exonid, exonlength, ExpDict[exonid])
        UntargetedSet = []
        for exonid in self.UntargetedSet:
            Untargetedset.append( (exonid, exonlegnth, ExpDict[exonid]) )
        self.TargetedSet = TargetedSet
        self.UntargetedSet = UntargetedSet
    def Permute(self):
        selected = set([])
        res_tar, res_untar = [], []
        i = 0
        while i < self.NTargeted:
            x = random.random()
            idx = bisect.bisect(self.cdf_vals, x)
            if x not in selected:
                selected.add(x)
                i += 1 
            else:
                continue

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result
def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]
def GetExonProb(exon_df):
    Total_length = sum(exon_df["exon length"].values)
    Probs = [float(x)/Total_length for x in exon_df["exon length"].values]
    return exon_df.index.values, Probs

def CollectDat(G1, G2):
    G1_pre, G1_post, G2_pre, G2_post = [], [], [], []
    for gene in G1.Genes:
        if gene not in G1.expdict:
            continue
        for ExonID in G1.Genes[gene].TargetedExons:
            pre, post = [], []
            for i, stage in enumerate(Stages):
                if i < 6:
                    pre.extend(G1.expdict[gene][ExonID][stage])
                else:
                    post.extend(G1.expdict[gene][ExonID][stage])
            #print pre, post
            #G1_pre.append(np.mean(pre))
            #G1_post.append(np.mean(post))
            G1_pre.extend(pre)
            G1_post.extend(post)
    for gene in G2.Genes:
        if gene not in G2.expdict:
            continue
        for ExonID in G2.Genes[gene].TargetedExons:
            pre, post = [], []
            for i, stage in enumerate(Stages):
                if i < 6:
                    pre.extend(G2.expdict[gene][ExonID][stage])
                else:
                    post.extend(G2.expdict[gene][ExonID][stage])
            #G2_pre.append(np.mean(pre))
            #G2_post.append(np.mean(post))
            G2_pre.extend(pre)
            G2_post.extend(post)
    return G1_pre, G1_post, G2_pre, G2_post

def load_mutation_exp(VarFile, expdictfile, outname="tmp.targeted.exons.xlsx", row_meta=None):
    row_meta = ins.AssignVar2Exon5(row_meta, VarFile, IntersectionWithPredicted=False)
    row_meta_with_gene = row_meta[row_meta["GeneHited"]=="T"]
    row_meta_with_gene.to_excel(outname, index=False)
    print ("Num.of.Genes:", row_meta_with_gene.groupby('gene_symbol').count().shape)

    expdict = ins.LoadGeneSetDataFromFil(expdictfile)
    Genes = list(set(row_meta_with_gene["gene_symbol"].values))
    gene_exon_set = GeneExonSet(expdict)
    for i, gene in enumerate(Genes):
        gene_df = row_meta_with_gene[row_meta_with_gene["gene_symbol"]==gene]
        TargetedExon = map(int, gene_df[gene_df["Vars"]!=""]["row_num"])
        UntargetedExon = map(int, gene_df[gene_df["Vars"]==""]["row_num"])
        TargetedExonLength = map(int, gene_df[gene_df["Vars"]!=""]["cds length"])
        UntargetedExonLength = map(int, gene_df[gene_df["Vars"]==""]["cds length"])
        gene_exon_set.addGene(gene, TargetedExon, UntargetedExon, TargetedExonLength, UntargetedExonLength)
    print (len(gene_exon_set.genes), len(gene_exon_set.TargetedExon),len(gene_exon_set.UntargetedExon))
    return gene_exon_set

#[("proband LGD", 'red', prolgd), ("gnomAD LGD", 'blue', gnomADlgd),
# ("gnomAD male LGD", 'yellow', gnomADlgd_male),("gnomAD female LGD", 'green', gnomADlgd_female) ]

def plot_mutation_exp(title, datasets, ylims=((4,12.25))):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    plt.title(title)
    for i, (title, color, Dat) in enumerate(datasets):
        Tseq, Terr, Useq, Uerr, All, Allerr = Dat
        ax.errorbar(range(2,14), Tseq, yerr = Terr, label=title, color=color)
        #ax.errorbar(range(2,14), Useq, yerr = Uerr, label="Exons without LGD mutations", color=color)
        ax.errorbar(range(2,14), All, yerr = Allerr, linestyle = '--', label="All Exons", color=color)

    ax.grid(True)
    ax.axvline(x=7.5, color="grey", linestyle="--")
    plt.xticks(np.arange(2,14), ins.Descriptions, rotation=60)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='upper right', fontsize=8)
    plt.xlabel("Dev Stages")
    plt.ylabel("Expression")
    plt.ylim(ylims)
    plt.show()

def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        #dic.update({col : sorted(df[col])})
        dic.update({col : np.sort(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

class HBT_DATA:
    def __init__(self):
        self.Stages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 
    def extract_matrix_table(self, File, OutName):
        hand = open(File, 'rb')
        out = open(OutName, 'wb')
        flag_out = False
        for l in hand:
            if flag_out:
                out.write(l)
            if l.strip() == "!series_matrix_table_begin":
                flag_out = True
            if l.strip() == "!series_matrix_table_end":
                flag_out = False
                break
        hand.close()
        out.close()
    def get_char_ch1(self, string):
        return string.strip('"').split()[0]
    def extract_col_info(self, File, OutName):
        hand = open(File, 'rb')
        csv_writer = csv.writer(open(OutName, 'wb'), delimiter=",")
        for l in hand:
            llist = l.strip().split("\t")
            label, data = llist[0], llist[1:]
            if label == "!Sample_geo_accession":
                sample_id = [x.strip('"') for x in data]
            if label == "!Sample_source_name_ch1":
                source_name = [x.strip('"').split("_")[0] for x in data]
            if label == "!Sample_characteristics_ch1":
                if self.get_char_ch1(data[0]) == "region:":
                    region = [x.strip('"').split()[1] for x in data]
                elif self.get_char_ch1(data[0]) == "Sex:":
                    sex = [x.strip('"').split()[1] for x in data]
                elif self.get_char_ch1(data[0]) == "age:":
                    age = [x.strip('"').split()[1] for x in data]
                elif self.get_char_ch1(data[0]) == "Stage:":
                    stage = [x.strip('"').split()[1] for x in data]
            if label == "!Sample_description":
                sample_desc = [x.strip('"') for x in data]
            if label == "!series_matrix_table_begin":
                break
        csv_writer.writerow(["column_num", "sample_id", "donor_id", "region", "sex", "age", "stage", "sample_desc"])
        for i, (a, b, c, d, e, f, g) in enumerate(zip(sample_id, source_name, region, sex, age, stage, sample_desc)):
            csv_writer.writerow([i+1,a,b,c,d,e,f,g])
    def extract_row_info(self, File, OutName, platform):
        hand = open(File, 'rb')
        out = csv.writer(open(OutName, 'wb'), delimiter=",")
        flag_out = False
        plat = None
        for l in hand:
            if l.strip() == "^PLATFORM = "+platform:
                plat = platform
            if flag_out and plat == platform:
                out.writerow(l.split("\t"))
            if l.strip() == "!platform_table_begin":
                flag_out = True
            if l.strip() == "!platform_table_end":
                flag_out = False
                break
        hand.close()
        return
    def LookALLMutationTargetedGenesHBT(self, Genes, structure_acronyms, GeneDat, smooth=True, drop_low_exp=True, fontsize=6, ylim=None):
        GeneExp, GeneRow, GeneCol = GeneDat
        SelectedGenes = GeneRow[GeneRow["gene_symbol"].isin(Genes)]
        gene_ids = list(SelectedGenes.index)
        #print gene_ids
        plt.close('all')
        stages = {}
        for i, gene_id in enumerate(gene_ids):
            sys.stdout.write("\r{}".format(i))
            #print gene_id
            #print GeneExp.loc[gene_id, :]
            for structure_acronym in structure_acronyms:
                #print structure_acronym
                region_meta = GeneCol[GeneCol["region"]==structure_acronym]
                for index, row in region_meta.iterrows():
                    Period = row["stage"]
                    if Period not in stages:
                        stages[Period] = []
                    stages[Period].append(GeneExp.get_value(gene_id, row["sample_id"]))
        seq = []
        for stage in self.Stages:
            try:
                seq.append(np.mean(stages[stage]))
            except KeyError:
                seq.append(0)
        print (seq)
        if smooth:
            seq = smooth_func(seq)
        fig, ax = plt.subplots(dpi=200)
        plt.title("TargetedExons over Region:{}".format(",".join(structure_acronyms)))
        ax.plot(xrange(1,16),seq)
        ax.grid(True)
        ax.axvline(x=7.5)
        #plt.xticks(np.arange(2,14), Stages, rotation=20)
        #ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        if ylim != None:
            plt.ylim(ylim)
        plt.show()

    class DevStageExp:
        def __init__(self, DevStage):
            self.DevStage = DevStage
            self.exps = np.array([])
            self.mean_exp = 0
        def add_exp(self, value):
            self.exps = np.append(self.exps, float(value))
            #self.mean_exp = round(np.mean(self.exps), 4)
        def get_nonzero_mean(self):
            try:
                self.non_zero_exps = [x for x in self.exps if x]
                return np.mean(self.non_zero), np.var(self.non_zero), np.median(self.non_zero_exps)
            except ZeroDivisionError:
                return 0, 0, 0

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import random
import bisect
import collections
class PredictIQFromExon:
    def __init__(self, VarFil, FamINFO, matrix):
        self.matrix = matrix
        self.df_var = pd.read_excel(VarFil)
        self.df_fam = pd.read_excel(FamINFO)
    def sameExon(self, var1, var2, gene, row_meta):
        gene_df = row_meta[row_meta["gene_symbol"]==gene]
        pos_1 = int(var1.split(":")[1])
        pos_2 = int(var2.split(":")[1])
        exon_1, exon_2 = None, None
        for i, row in gene_df.iterrows():
            start = int(row["start"]) - 2
            end = int(row["end"]) + 2
            if pos_1 >= start and pos_1 <= end:
                exon_1 = i
            if pos_2 >= start and pos_2 <= end:
                exon_2 = i
        return (exon_1 == exon_2) and (exon_1 != None) and (exon_2 != None)
    def get_exon_id(self, var, gene, row_meta):
        gene_df = row_meta[row_meta["gene_symbol"]==gene]
        pos = int(var.split(":")[1])
        for i, row in gene_df.iterrows():
            start = int(row["start"]) - 2
            end = int(row["end"]) + 2
            if pos >= start and pos <= end:
                return i
        return None
    def isLastExon(self, exon_id, row_meta):
        gene = row_meta.get_value(exon_id, "gene_symbol")
        next_gene = row_meta.get_value(exon_id+1, "gene_symbol")
        return not (gene == next_gene)
        def cdf(self, weights):
            total = sum(weights)
            result = []
            cumsum = 0
            for w in weights:
                cumsum += w
                result.append(cumsum / total)
            return result
    def choice(self, population, weights):
        assert len(population) == len(weights)
        cdf_vals = cdf(weights)
        x = random.random()
        idx = bisect.bisect(cdf_vals, x)
        return population[idx]
    def GetExonProb(self, exon_df):
        Total_length = sum(exon_df["exon length"].values)
        Probs = [float(x)/Total_length for x in exon_df["exon length"].values]
        return exon_df.index.values, Probs
    def avgseq(self, seq, start, end):
        return np.mean(seq[start: end])
    def getIQ(self, famID, which="probandNVIQ"):
        return float(self.df_fam[self.df_fam["familyId"] == famID][which].values[0])
    def regGene(self, dat):
        regr = linear_model.LinearRegression(fit_intercept=False)
        X_train = np.array([[x[4]] for x in dat])
        y_train = np.array([x[3] for x in dat])
        regr.fit(X_train, y_train)
        return regr
    def regGene2(self, X, Y):
        regr = linear_model.LinearRegression(fit_intercept=False)
        X = X.reshape(-1,1)
        regr.fit(X, Y)
        return regr
    def plotGene(self, gene, regr, dat):
        X_train = np.array([[x[4]] for x in dat])
        y_train = np.array([x[3] for x in dat])
        y_pred = regr.predict(X_train)
        #print('Coefficients:', regr.coef_)
        R, P = scipy.stats.pearsonr([x[0] for x in X_train], y_train)
        plt.title("Gene:{}, R:{}, P:{}".format(gene, R,P))
        plt.xlim(0,3)
        plt.ylim(0, max(y_train+y_pred))
        plt.scatter(X_train, y_train,  color='black')
        plt.plot(np.append(X_train, [0]), np.append(y_pred, 0), color='blue', linewidth=3)
        plt.show()
    def model(self, df_row, var2leave = None, plot=True):
        self.gene2slope = {}
        ALL_Exp = np.array([])
        ALL_Normed_IQD = np.array([])
        Genes = list(set(self.df_var["effectGene"].values))
        for gene in Genes:
            tmp_df = self.df_var[self.df_var["effectGene"]==gene]
            famids = tmp_df["familyId"].values
            varids = tmp_df["location"].values
            if var2leave != None:
                varids = [x for x in varids if x!=var2leave]
            exonids = [self.get_exon_id(x, gene, df_row) for x in varids]
            exonids = [x for x in exonids if x != None]
            #print exonids
            exonids = [x for x in exonids if not self.isLastExon(x, df_row)]
            exps = []
            for exonid in exonids:
                seq = self.matrix.iloc[exonid-1, 0:12].values
                #print seq
                exp = self.avgseq(seq, 0, 12) / 10
                exps.append(exp)
            IQs = [max(0, (100-self.getIQ(x))) for x in famids]
            #exonids = [x for x in exonids if x!=None]
            dat = zip(famids, varids, exonids, IQs, exps)
            dat = [x for x in dat if x[2]!= None]
            exps = np.array([[x[4]] for x in dat])
            IQDs = np.array([x[3] for x in dat])
            if dat == []:
                continue
            regr = self.regGene(dat)
            slope = regr.coef_[0]
            if slope == 0:
                continue
            self.gene2slope[gene] = slope
            ALL_Exp = np.append(ALL_Exp, exps)
            NormIQDs = np.array([x[3]/slope for x in dat])
            ALL_Normed_IQD = np.append(ALL_Normed_IQD, NormIQDs)
        model = linear_model.LinearRegression(fit_intercept=False)
        ALL_Exp = ALL_Exp.reshape(-1,1)
        R, P = scipy.stats.pearsonr([x[0] for x in ALL_Exp], ALL_Normed_IQD)
        if plot:
            plt.title("R:{}, P:{}".format(R,P))
            model.fit(ALL_Exp, ALL_Normed_IQD)
            plt.scatter(ALL_Exp, ALL_Normed_IQD)
            upper = max(ALL_Normed_IQD)
            plt.plot([0, upper], [0,model.predict(upper)], color='red')
            plt.show()
        else:
            print ("R:{}, P:{}".format(R,P)) 
    def LoadData(self, df_row, method = 'rolling_time', time_start=0, time_end=12):
        assert time_end > time_start
        self.time_start, self.time_end = time_start, time_end
        Genes = list(set(self.df_var["effectGene"].values))
        self.VarList = {}
        self.Gene2Var = {}
        self.Genes = []
        for gene in Genes:
            tmp_df = self.df_var[self.df_var["effectGene"]==gene]
            famids = tmp_df["familyId"].values
            varids = tmp_df["location"].values
            exonids = [self.get_exon_id(x, gene, df_row) for x in varids]
            exonids = [x for x in exonids if x != None]
            exonids = [x for x in exonids if not self.isLastExon(x, df_row)]
            exps = [] 
            for exonid in exonids:
                #seq = self.matrix.iloc[exonid-1, 0:12].values
                seq = self.matrix.iloc[exonid-1, 0:12].values
                if method == 'rolling_time':
                    exp = self.avgseq(seq, self.time_start, self.time_end)
                elif method == 'max':
                    exp = max(seq) 
                elif method == 'min':
                    exp = min(seq)
                exps.append(exp)
            IQs = [max(0, (100-self.getIQ(x))) for x in famids]
            dat = zip(famids, varids, exonids, IQs, exps)
            dat = [x for x in dat if x[2]!= None]
            if len(dat) < 2:
                continue
            self.Genes.append(gene)
            for a in dat:
                var = variant(a[1], a[0], a[3], gene, a[2], a[4])
                self.VarList[var.VarID] = var
                if gene in self.Gene2Var:
                    self.Gene2Var[gene].append(var)
                else:
                    self.Gene2Var[gene] = [var]
    def model2(self, plot=True, var2leave = None):
        self.gene2slope = {}
        ALL_Exp = np.array([])
        ALL_Normed_IQD = np.array([])
        for gene in self.Genes:
            Vars = self.Gene2Var[gene]
            if var2leave != None:
                Vars = [x for x in Vars if x.VarID != var2leave]
            exps = np.array([x.ExonExp for x in Vars])
            IQDs = np.array([x.ProbandIQ for x in Vars])
            regr = self.regGene2(exps, IQDs)
            slope = regr.coef_[0]
            #print gene, slope
            if slope == 0:
                continue
            self.gene2slope[gene] = slope
            ALL_Exp = np.append(ALL_Exp, exps)
            NormIQDs = np.array([x/slope for x in IQDs])
            ALL_Normed_IQD = np.append(ALL_Normed_IQD, NormIQDs)
        #print ALL_Exp, ALL_Normed_IQD
        model = linear_model.LinearRegression(fit_intercept=False)
        ALL_Exp = ALL_Exp.reshape(-1,1)
        R, P = scipy.stats.pearsonr([x[0] for x in ALL_Exp], ALL_Normed_IQD)
        if plot:
            plt.title("R:{}, P:{}, time{}-{}".format(round(R,6),round(P,6), self.time_start, self.time_end))
            model.fit(ALL_Exp, ALL_Normed_IQD)
            plt.scatter(ALL_Exp, ALL_Normed_IQD)
            upper = max(ALL_Normed_IQD)
            plt.plot([0, upper], [0,model.predict(upper)], color='red')
            plt.show()
    def Predict(self, var):
        return self.gene2slope[var.Gene] * var.ExonExp

class variant:
    def __init__(self, VarID, ProbandID, ProbandIQ, Gene, ExonID, ExonExp):
        self.VarID = VarID
        self.ProbandID = ProbandID
        self.ProbandIQ = ProbandIQ
        self.Gene = Gene
        self.ExonID = ExonID
        self.ExonExp = ExonExp
    def show(self):
        print ("VarID:{} FamID:{} IQ:{} Gene:{} ExonID:{} ExonExp:{}".format(
        self.VarID, self.ProbandID, self.ProbandIQ, self.Gene, self.ExonID, self.ExonExp))

def loaddict():
    res = {}
    fin = open("/Users/jiayao/Work/BrainDisorders/src/cds.dict", 'rt')
    for l in fin:
        llist = l.strip().split()
        gene, exon_s, cds_s, cds_e = llist[0], int(llist[1])-1, int(llist[2])-1, int(llist[3])
        if gene not in res:
            res[gene] = {}
        if exon_s not in res[gene]:
            res[gene][exon_s] = (cds_s, cds_e)
    return res
def addcds(row, cds_dict):
    s = row["start"]
    gene = row["gene_symbol"]
    if gene not in cds_dict:
        return row["exon length"]
    if s in cds_dict[gene]:
        cds_s, cds_e = cds_dict[gene][s]
        return cds_e - cds_s
    else:
        return row["exon length"]


def regGene(X, Y):
    regr = linear_model.LinearRegression(fit_intercept=False)
    X_train = np.array(X)
    X_train = np.reshape(X_train, (-1, 1))
    y_train = np.array(Y)
    y_train = np.reshape(y_train, (-1, 1))
    regr.fit(X_train, y_train)
    return regr
def plotGene(gene, regr, dat):
    X_train = np.array([[x[4]] for x in dat])
    y_train = np.array([x[3] for x in dat])
    y_pred = regr.predict(X_train)
    #print('Coefficients:', regr.coef_)
    R, P = scipy.stats.pearsonr([x[0] for x in X_train], y_train)
    plt.title("Gene:{}, R:{}, P:{}".format(gene, R,P))
    plt.xlim(0,3)
    plt.ylim(0, max(y_train+y_pred))
    plt.scatter(X_train, y_train,  color='black')
    plt.plot(np.append(X_train, [0]), np.append(y_pred, 0), color='blue', linewidth=3)
    plt.show()

def SSE(List1, List2):
    #return sum([(x-y)**2 for x,y in zip(List1, List2)])
    #return scipy.stats.mstats.gmean([(x-y)**2 for x,y in zip(List1, List2)]) 
    return np.mean([(x-y)**2 for x,y in zip(List1, List2)]) 
def PredErrMean(List1, List2):
    return np.mean([abs(x-y) for x,y in zip(List1, List2)])
def PredErrMedian(List1, List2):
    return np.median([abs(x-y) for x,y in zip(List1, List2)])
def PredErrgMean(List1, List2):
    return scipy.stats.mstats.gmean([abs(x-y) for x,y in zip(List1, List2)])

def CrossVal(df, Splits, Xlabels, Ylabel="NVIQ", Fold=5, Intercept=False, Error="median", N=100):
    SCORES_TRAIN, SCORES_TEST = [], []
    for i in range(N):
        df = df.sample(frac=1).reset_index(drop=True)
        Scores_Train, Scores_Test = [], []
        for i in range(Fold):
            heldout = Splits[i]
            TestingDat = df.loc[heldout[0]:heldout[1],:]
            TrainingDat = df[~df["KEY"].isin(TestingDat["KEY"].values)]
            assert len(Xlabels) >= 1
            X_train = np.reshape(np.array(TrainingDat[Xlabels[0]].values), (-1,1))
            if Intercept:
                const_train = np.reshape(np.ones(X_train.shape[0]),(-1,1))
                X_train = np.hstack((const_train, X_train))
            X_test = np.reshape(np.array(TestingDat[Xlabels[0]].values), (-1,1))
            #const_test = np.reshape(np.ones(X_test.shape[0]),(-1,1))
            #X_test = np.hstack((const_test, X_test))
            for i in range(1, len(Xlabels)):
                x_train = np.reshape(np.array(TrainingDat[Xlabels[i]].values), (-1,1))
                X_train = np.hstack((X_train, x_train))
                x_test = np.reshape(np.array(TestingDat[Xlabels[i]].values), (-1,1))
                X_test = np.hstack((X_test, x_test))
            Y_train = np.reshape(np.array(TrainingDat["NVIQ"].values), (-1, 1))
            Y_test = np.reshape(np.array(TestingDat["NVIQ"].values), (-1, 1))
            glm = sm.GLM(Y_train, X_train, family=sm.families.Gaussian())
            res = glm.fit(method="newton")
            pred_train = res.predict(X_train)
            pred_test = res.predict(X_test)
            if Error == "mean":
                Scores_Train.append(PredErrMean(pred_train, Y_train))
                Scores_Test.append(PredErrMean(pred_test, Y_test))
            else:
                Scores_Train.append(PredErrMedian(pred_train, Y_train))
                Scores_Test.append(PredErrMedian(pred_test, Y_test))
        SCORES_TRAIN.append(np.mean(Scores_Train))
        SCORES_TEST.append(np.mean(Scores_Test))
    return SCORES_TRAIN, SCORES_TEST

def DosageModel(df, SameGender=False, plot=True):
    GeneCount = df.groupby("effectGene")["effectGene"].count()
    df["GeneCount"] = df.apply(lambda row: GeneCount[row["effectGene"]], axis=1)
    df = df[df["GeneCount"]>=2].copy()
    IQ_diff_dosage, IQ_diff_gene, IQ_diff_mean = [], [], []
    IIQs = []
    pred_dosage, pred_gene, pred_mean = [],[],[]
    #avg_IQ = np.mean(df["NVIQ"].values)
    ALL_IQ = []
    for i, row in df.iterrows():
        familyId, gene, ralexp, IQ, gender= row["familyId"], row["effectGene"], row["Rel.exp.amean"], row["NVIQ"], row["gender"]
        if SameGender:
            tmp = df[(df["effectGene"]==gene) & (df["familyId"]!=familyId) & (df["gender"]==gender)]
        else:
            tmp = df[(df["effectGene"]==gene) & (df["familyId"]!=familyId)]
        if tmp.shape[0] < 1:
            continue
        ALL_IQ.append(row["NVIQ"])
    avg_IQ = np.mean(ALL_IQ)
    N = 0
    for i, row in df.iterrows():
        familyId, gene, ralexp, IQ, gender= row["familyId"], row["effectGene"], row["Rel.exp.amean"], row["NVIQ"], row["gender"]
        if SameGender:
            tmp = df[(df["effectGene"]==gene) & (df["familyId"]!=familyId) & (df["gender"]==gender)]
        else:
            tmp = df[(df["effectGene"]==gene) & (df["familyId"]!=familyId)]
        if tmp.shape[0] < 1:
            continue
        IQs = tmp["NVIQ"].values
        IQDiffs = [max(0, (100-x)) for x in IQs]
        rel_exps = list(tmp["Rel.exp.amean"].values)
        regr = regGene(rel_exps, IQDiffs)
        slope = regr.coef_[0][0]
        IQpre_dosage = max(0, (100 - slope * ralexp))
        IQpre_gene = np.mean(IQs)
        if abs(IQ - IQpre_dosage) > 40:
            if plot:
                print(row["KEY"], gene, IQ, IQpre_dosage, ralexp)
            #continue
        df.loc[i, "Dosage"] = IQpre_dosage
        IQ_diff_dosage.append(abs(IQ - IQpre_dosage))
        IQ_diff_gene.append(abs(IQ - IQpre_gene))
        IQ_diff_mean.append(abs(IQ - avg_IQ))
        pred_dosage.append(IQpre_dosage)
        pred_gene.append(IQpre_gene)
        pred_mean.append(avg_IQ)
        IIQs.append(IQ)
        N += 1
    if plot:
        print(N)
        plt.figure(figsize=(4,4), dpi=120)
        plt.boxplot([IQ_diff_dosage, IQ_diff_gene, IQ_diff_mean], labels = ["dosamge", "gene", "mean"])
        plt.ylabel("Nonverbal IQ difference")
        plt.grid(True)
        plt.show()
        print(scipy.stats.mannwhitneyu(IQ_diff_dosage, IQ_diff_mean, alternative='less'))
        print(scipy.stats.mannwhitneyu(IQ_diff_dosage, IQ_diff_gene, alternative='less'))
        print("Mean:\tDosage:%.3f\tGene:%.3f\tMean:%.3f"%(np.mean(IQ_diff_dosage), np.mean(IQ_diff_gene), np.mean(IQ_diff_mean)))
        print("Median:\tDosage:%.3f\tGene:%.3f\tMean:%.3f"%(np.median(IQ_diff_dosage), np.median(IQ_diff_gene), np.median(IQ_diff_mean)))
    #return df, IQ_diff_dosage, IQ_diff_gene, IQ_diff_mean
    #print(np.mean(ALL_IQ), np.mean(IIQs), avg_IQ)
    return df, pred_dosage, pred_gene, pred_mean, ALL_IQ


def csv2vcf(infname, outfname):
    reader = csv.reader(open(infname, "rt"))
    writer = csv.writer(open(outfname, "wt"), delimiter="\t")
    writer.writerow(["##fileformat=VCFv4.2"])
    writer.writerow(["#CHROM", "POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"])
    head = next(reader)
    idx_vcfVariant = head.index("vcfVariant") 
    idx_key = head.index("KEY")
    for row in reader:
        vcfVariant = row[idx_vcfVariant]
        Chr, Pos, Ref, Alt = vcfVariant.split(":")
        llist = []
        llist.append(Chr)
        llist.append(Pos)
        llist.append(row[idx_key])
        llist.append(Ref)
        llist.append(Alt)
        llist.append(".")
        llist.append(".")
        llist.append(".")
        llist.append(".")
        writer.writerow(llist)

##############################################################################
# Parse GTF format for gene/transcripts/exons
##############################################################################
class GTFRecord:
    def __init__(self, Chr, source, Type, start, end, strand, info):
        self.Chr = Chr
        self.source = source
        self.Type = Type
        self.start = start
        self.end = end
        self.strand = strand
        self.info = info

class GTFGene:
    def __init__(self, GeneName, GeneID, strand):
        self.GeneName = GeneName
        self.GeneID = GeneID
        self.strand = strand
        self.Transcripts = {}

class GTFTranscript:
    def __init__(self, gene, TranscriptID, TranscriptName, strand):
        self.TranscriptID = TranscriptID
        self.TranscriptName = TranscriptName
        self.GeneName = gene
        self.Exons = {}
        self.strand = strand
    def SortExons(self):
        self.ExonSeq = []
        if self.strand == "+":
            self.ExonSeq = sorted(self.Exons.values(), key=lambda x:x.start)
        else:
            self.ExonSeq = sorted(self.Exons.values(), key=lambda x:x.start, reverse=True)
    def LastExonJunction(self): # Interval of last exon and 2nd-last 55nt EEJ, tuple of tuple
        if self.strand == "+":
            LEJ = (self.ExonSeq[-2].end - 55, self.ExonSeq[-2].end)
            LE = (self.ExonSeq[-1].start, self.ExonSeq[-1].end)
        elif self.strand == "-":
            LEJ = (self.ExonSeq[-2].start, self.ExonSeq[-2].start + 55)
            LE = (self.ExonSeq[-1].start, self.ExonSeq[-1].end)
        return (LE, LEJ)

class GTFExon:
    def __init__(self, exon_id, start, end, TranscriptID, strand):
        self.ExonID = exon_id
        self.TranscriptID = TranscriptID
        self.start = int(start)
        self.end = int(end)
        self.strand = strand
        self.cds = None

class GTFCDS:
    def __init__(self, exon_id, start, end, TranscriptID, strand):
        self.ExonID = exon_id
        self.TranscriptID = TranscriptID
        self.start = int(start)
        self.end = int(end)
        self.strand = strand

def gtf_info_parser(info):
    res = {}
    for term in info.split(";"):
        if term == "":
            continue
        # print(">",term)
        key, v = term.split()
        v = v.strip('"')
        res[key] = v
    return res

def LoadGeneCode(genecodefil):
    Genes = {}
    Transcripts = {}
    hand = open(genecodefil, 'rt')
    for l in hand:
        if l.startswith("#"):
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        if llist[2] == "gene":
            Genes[info["gene_name"]] = GTFRecord(
                llist[0], llist[1], llist[2], llist[3], llist[4], llist[6], info)
            Transcripts[info["gene_name"]] = []
        elif llist[2] == "transcript":
            if info["gene_name"] not in Genes:
                Genes[info["gene_name"]] = GTFRecord(
                    llist[0], llist[1], llist[2], llist[3], llist[4], llist[6], info)
                Transcripts[info["gene_name"]] = []
            Transcripts[info["gene_name"]].append(
                GTFRecord(llist[0], llist[1], llist[2], llist[3], llist[4], llist[6], info))
    return Genes, Transcripts

def isLEJ(Gene, Pos, Ref, Alt, Genes):
    Pos, LenV = int(Pos), len(Ref)-len(Alt)
    gene_obj = Genes[Gene]
    count1, count2, count3 = 0, 0, 0
    for transid, transobj in gene_obj.Transcripts.items():
        transobj.SortExons()
        if len(transobj.Exons) >= 2:
            count1 += 1
            interval1, interval2 = transobj.LastExonJunction()
            if (Pos > interval1[0] and Pos < interval1[1]):
                count2 += 1
            elif(Pos > interval2[0] and Pos < interval2[1]):
                count3 += 1
    if count2 != 0:
        isle = 'T'
    else:
        isle = "F"
    if count3 != 0:
        islej = 'T'
    else:
        islej = 'F'
    return "{}/{}".format(count2+count3, count1), isle, islej

def subsetGTF(GTF, GENESET, outfile):
    gtf = GTF
    out = open(outfile, 'wt')
    hand = open(gtf, 'rt')
    for l in hand:
        if l.startswith("#"):
            out.write(l)
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        if info["gene_name"] in GENESET:
            out.write(l)

##############################################################################
# Variance Explained for GLM and same exon prediction
##############################################################################

def VarianceExplained(Predictions, TrueValues):
    TSS = sum([(x-y)**2 for x,y in zip(TrueValues, [np.mean(TrueValues)]*len(TrueValues))])
    SSR = sum([(x-y)**2 for x,y in zip(TrueValues, Predictions)])
    #print(SSR, TSS)
    R2_mean = 1- SSR/TSS
    return R2_mean

def SameExonPred(df, target="NVIQ"):
    ExonCount = df.groupby("ExonID")["ExonID"].count()
    df["ExonCount"] = df.apply(lambda row: ExonCount[row["ExonID"]], axis=1)
    SameExon = df[df["ExonCount"]>=2]
    SameExonPredictErr = []
    SameExonPredictions = []
    SameExonTarget = []
    new_dfs = []
    for i, row in SameExon.iterrows():
        ExonID = row["ExonID"]
        #subdf = SameExon[(SameExon["ExonID"]==ExonID) & (SameExon["familyId"]!=row["familyId"])
        #               & (SameExon["gender"]==row["gender"])]
        subdf = SameExon[(SameExon["ExonID"]==ExonID) & (SameExon["familyId"]!=row["familyId"])]
        if subdf.shape[0] == 0:
            continue
        new_dfs.append(row)
        _Pred = np.mean(SameExon[(SameExon["ExonID"]==ExonID)&(SameExon["familyId"]!=row["familyId"])][target].values)
        _Target = row[target]
        SameExonPredictErr.append(abs(_Pred-_Target))
        SameExonPredictions.append(_Pred)
        SameExonTarget.append(_Target)
    NewSameExon = pd.DataFrame(new_dfs)
    return SameExonPredictions, SameExonTarget, SameExonPredictErr, NewSameExon

def llllll(Intercept, AllDF, TestingDF, aim="NVIQ"):
    TestingDat = TestingDF
    TrainingDat = AllDF[~AllDF["KEY"].isin(TestingDat["KEY"].values)]
    Xlabels = ["ExonPrenatalExp.amean", "ExonPostnatalExp.amean", "phyloP100way", "Functional", "DOM.TRUNC"]
    X_train = np.reshape(np.array(TrainingDat[Xlabels[0]].values), (-1,1))
    X_test = np.reshape(np.array(TestingDat[Xlabels[0]].values), (-1,1))
    if Intercept:
        const_train = np.reshape(np.ones(X_train.shape[0]),(-1,1))
        X_train = np.hstack((const_train, X_train))
        const_test = np.reshape(np.ones(X_test.shape[0]),(-1,1))
        X_test = np.hstack((const_test, X_test))
    for i in range(1, len(Xlabels)):
        x_train = np.reshape(np.array(TrainingDat[Xlabels[i]].values), (-1,1))
        X_train = np.hstack((X_train, x_train))
        x_test = np.reshape(np.array(TestingDat[Xlabels[i]].values), (-1,1))
        X_test = np.hstack((X_test, x_test))
    Y_train = np.reshape(np.array(TrainingDat[aim].values), (-1, 1))
    Y_test = np.reshape(np.array(TestingDat[aim].values), (-1, 1))
    glm = sm.GLM(Y_train, X_train, family=sm.families.Gaussian())
    res = glm.fit()
    pred_train = res.predict(X_train)
    #print(X_test.shape, X_train.shape)
    pred_test = res.predict(X_test)
    GLM_err = [abs(x-y) for x,y in zip(pred_test, Y_test)]
    GLM_err = [x[0] for x in GLM_err]
    return res, GLM_err, pred_test, Y_test, pred_train, Y_train

def PlotScatter4R2(SEP, SET, GLMP, GLMT, title):
    fig, axs = plt.subplots(1,2, figsize=(9,4))
    axs[0].scatter(SEP, SET)
    axs[0].plot([0,100],[0,100], color="black")
    axs[0].text(10, 90, "r=%.3f p=%.2e"%(pearsonr(SEP, SET)))
    axs[0].text(10, 85, "R2=%.3f"%VarianceExplained(SEP, SET))
    axs[1].scatter(GLMP, GLMT)
    axs[1].plot([0,100],[0,100], color="black")
    axs[1].text(10, 90, "r=%.3f p=%.2e"%(pearsonr([x for x in GLMP], [x[0] for x in GLMT])))
    axs[1].text(10, 85, "R2=%.3f"%VarianceExplained(GLMP, GLMT))
    axs[0].set_ylabel(title)
    axs[0].set_xlabel("Same Exon predicted %s"%title)
    #axs[1].set_ylabel("NVIQ")
    axs[1].set_xlabel("GLM predicted %s"%title)
    plt.show()

def PredERRORwithP():
    plt.figure(figsize=(4,4), dpi=120)
    plt.boxplot([SameExonNVIQPredictErr, GLM_err1, GLM_err2], labels = ["same exom", "GLM w/ b0", "GLM w/o b0"])

    t1, p1 = scipy.stats.wilcoxon(SameExonNVIQPredictErr, GLM_err1)
    t2, p2 = scipy.stats.mannwhitneyu(SameExonNVIQPredictErr, GLM_err1, alternative='less')
    x1, x2 = 1,2
    y,h =35, 2
    col='black'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "p=%.3e(mwu)"%(p2), ha='center', va='bottom', color=col)
    plt.text((x1+x2)*.5, y+h, "p=%.3e(wilcoxon)"%(p1/2), ha='center', va='bottom', color=col)

    t1, p1 = scipy.stats.wilcoxon(SameExonNVIQPredictErr, GLM_err2)
    t2, p2 = scipy.stats.mannwhitneyu(SameExonNVIQPredictErr, GLM_err2, alternative='less')
    x1, x2 = 1,3
    y,h =60, 2
    col='black'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "p=%.3e(mwu)"%(p2), ha='center', va='bottom', color=col)
    plt.text((x1+x2)*.5, y+h, "p=%.3e(wilcoxon)"%(p1/2), ha='center', va='bottom', color=col)

    plt.ylabel("Nonverbal IQ difference")
    plt.grid(True)
    plt.title("NVIQ prediction error")
    plt.show()


##############################################################################
# GTEx 
##############################################################################

def PreProcessTranscripTPM(SelectedSamples, SelectedGenes):
    csv.field_size_limit(sys.maxsize)
    reader = csv.reader(open("../data/GTEx/GTEx_Analysis_2016-01-15_v7_RSEMv1.2.22_transcript_tpm.txt", 'rt'), delimiter="\t")
    writer = csv.writer(open("../data/GTEx/GTEx_180Indiv_transcript_tpm.txt", 'wt'), delimiter="\t")
    header = next(reader)
    indices = []
    new_header = header[:2]
    for i,xx in enumerate(header[2:]):
        #doner = xx.split("-")[1]
        #if doner in Doners:
        if xx in SelectedSamples:
            indices.append(i+2)
            new_header.append(xx)
    writer.writerow(new_header)
    for row in reader:
        new_row = [row[0].split(".")[0], row[1].split(".")[0]]
        if new_row[1] not in SelectedGenes:
            continue
        for i in indices:
            new_row.append(row[i])
        writer.writerow(new_row)

def PreProcessGeneTPM(SelectedSamples, SelectedGenes):
    csv.field_size_limit(sys.maxsize)
    #reader = csv.reader(open("../data/GTEx/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct", 'rt'), delimiter="\t")
    reader = csv.reader(open("../data/GTEx/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.gct", 'rt'), delimiter="\t")
    writer = csv.writer(open("../data/GTEx/GTEx_180Indiv_gene_rc.txt", 'wt'), delimiter="\t")
    header = next(reader)
    indices = []
    new_header = header[:2]
    for i,xx in enumerate(header[2:]):
        if xx in SelectedSamples:
            indices.append(i+2)
            new_header.append(xx)
    writer.writerow(new_header)
    for row in reader:
        new_row = [row[0].split(".")[0], row[1].split(".")[0]]
        if new_row[0] not in SelectedGenes:
            continue
        for i in indices:
            new_row.append(row[i])
        writer.writerow(new_row)

def PreProcessExonTPM(SelectedSamples, SelectedGenes):
    csv.field_size_limit(sys.maxsize)
    reader = csv.reader(open("../data/GTEx/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_exon_reads.csv", 'rt'), delimiter=",")
    writer = csv.writer(open("../data/GTEx/GTEx_180Indiv_exon_rc.txt", 'wt'), delimiter="\t")
    header = next(reader)
    indices = []
    new_header = header[:1]
    for i,xx in enumerate(header[1:-1]):
        #doner = xx.split("-")[1]
        #if doner in Doners:
        if xx in SelectedSamples:
            indices.append(i+1)
            new_header.append(xx)
    new_header.append(header[-1])
    indices.append(-1)
    writer.writerow(new_header)
    for row in reader:
        new_row = [row[0]]
        #print(row[-1])
        if row[-1].split(".")[0] not in SelectedGenes:
            continue
        for i in indices:
            new_row.append(row[i])
        writer.writerow(new_row)

def GTExTranscriptSimilarity(GTExTransTPM, GTEx_LGDs, GTExSample, EnsGeneID, Tissue):
    Gene_LGDs = GTEx_LGDs[GTEx_LGDs["SEVERE_GENE"]==EnsGeneID]
    WithLGDSamples = Gene_LGDs["INDV"].values
    WithLGDDoners = [x.split("-")[1] for x in WithLGDSamples]
    TissueWithLGD = GTExSample[(GTExSample["Doner"].isin(WithLGDDoners)) & (GTExSample["SMTS"]==Tissue)]
    TPMDat = GTExTransTPM[GTExTransTPM["gene_id"]==EnsGeneID]
    if TPMDat.shape[0] == 0:
        return None
    ExpDat = GTExSample[GTExSample["SMTS"]==Tissue]
    SameExon, DiffExon = [], []
    for row1, row2 in itertools.combinations(Gene_LGDs.iterrows(), r=2):
        try:
            row1, row2 = row1[1], row2[1]
            #doner1, doner2 = row1["INDV"].split("-")[1], row2["INDV"].split("-")[1]
            doner1, doner2 = row1["INDV"].split("-")[1], row2["INDV"].split("-")[1]
            #print(doner1, doner2)
            sp1 = ExpDat[ExpDat["Doner"]==doner1]["SAMPID"].values
            sp2 = ExpDat[ExpDat["Doner"]==doner2]["SAMPID"].values
            #print(sp1, sp2)
            if len(sp1) == 0 or len(sp2) == 0:
                continue
            V1, V2 = [], []
            for i, row in TPMDat.iterrows():
                exp1, exp2 = [], []
                for sp in sp1:
                    exp1.append(row[sp])
                for sp in sp2:
                    exp2.append(row[sp])
                exp1 = np.mean(exp1)
                exp2 = np.mean(exp2)
                V1.append(exp1)
                V2.append(exp2)
            #print(V1, V2)
            AngularDis = AngularDistance(V1, V2)
            if set(row1["Exons"].split(";")).intersection(set(row2["Exons"].split(";"))) != set([]):
                if AngularDis == AngularDis:
                    SameExon.append(AngularDis)
            else:
                if AngularDis == AngularDis:
                    DiffExon.append(AngularDis)
        except:
            #print(EnsGeneID)
            pass
    return SameExon, DiffExon

def AngularDistance(V1, V2):
    return (2/math.pi) * math.acos(1-scipy.spatial.distance.cosine(V1, V2))

def GTExGeneSimilarity(GTExGenesTPM, GTEx_LGDs, GTExSample, EnsGeneID, Tissue):
    # Compare exp diff between same exon / diff exon
    Gene_LGDs = GTEx_LGDs[GTEx_LGDs["SEVERE_GENE"]==Gene]
    WithLGDSamples = Gene_LGDs["INDV"].values
    TissueWithLGD = SelectedTissueSamples[(SelectedTissueSamples["Doner"].isin(WithLGDDoners)) &  (SelectedTissueSamples["SMTS"]==Tissue)]
    XXX = GeneTPM[GeneTPM["Description"]==EnsGeneID]
    if XXX.shape[0] == 0:
        return None
    ExpDat = SelectedTissueSamples[SelectedTissueSamples["SMTS"]==Tissue]
    Log2ExpOfGeneTissue = []
    for sp in ExpDat["SAMPID"].values:
        Log2ExpOfGeneTissue.append(XXX[sp])
    Log2ExpOfGeneTissue = [math.log2(x+1) for x in Log2ExpOfGeneTissue]
    GeneTissueMean = np.mean(Log2ExpOfGeneTissue)
    GeneTissueStd = np.std(Log2ExpOfGeneTissue)
    VarinGene = GTEx_LGDs[GTEx_LGDs["SEVERE_GENE_NAME"]==Gene]
    SameExon, DiffExon = [], []
    for row1, row2 in itertools.combinations(VarinGene.iterrows(), r=2):
        row1,row2 = row1[1], row2[1]
        doner1, doner2 = row1["INDV"].split("-")[1], row2["INDV"].split("-")[1]
        sp1 = ExpDat[ExpDat["Doner"]==doner1]["SAMPID"].values
        sp2 = ExpDat[ExpDat["Doner"]==doner2]["SAMPID"].values
        if len(sp1) == 0 or len(sp2) == 0:
            continue
        exp1, exp2 = [], []
        for sp in sp1:
            exp1.append(math.log2(XXX[sp]+1))
        for sp in sp2:
            exp2.append(math.log2(XXX[sp]+1))
        exp1 = (np.mean(exp1) - GeneTissueMean)/GeneTissueStd
        exp2 = (np.mean(exp2) - GeneTissueMean)/GeneTissueStd
        if row1["Exons"] == row2["Exons"]:
            SameExon.append(abs(exp1-exp2))
        else:
            DiffExon.append(abs(exp1-exp2))

def RPKM(ReadCount, Length, Libsize):
    ReadCount = float(ReadCount)
    return (ReadCount * 1e9) / (Libsize*Length)

def GeneRC2RPKM(LibsizeDict, GenesizeDict):
    reader = csv.reader(open("../data/GTEx/GTEx_180Indiv_gene_rc.txt", 'rt'), delimiter="\t")
    writer = csv.writer(open("../data/GTEx/GTEx_180Indiv_gene_rpkm.txt", 'wt'), delimiter="\t")
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        new_row = [row[0].split(".")[0], row[1] ]
        for i, SP in enumerate(header[2:]):
            libsize = LibsizeDict[SP]
            genesize = GenesizeDict[row[0].split(".")[0]]
            rpkm = round(RPKM(row[i+2], genesize, libsize), 2)
            new_row.append(rpkm)
        writer.writerow(new_row)

def ExonRC2RPKM(LibsizeDict, ExonsizeDict):
    reader = csv.reader(open("../data/GTEx/GTEx_180Indiv_exon_rc.txt", 'rt'), delimiter="\t")
    writer = csv.writer(open("../data/GTEx/GTEx_180Indiv_exon_rpkm.txt", 'wt'), delimiter="\t")
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        new_row = [row[0].split(".")[0] ]
        for i, SP in enumerate(header[1:-1]):
            libsize = LibsizeDict[SP]
            exonsize = ExonsizeDict[row[-1]]
            rpkm = round(RPKM(row[i+1], exonsize, libsize), 2)
            new_row.append(rpkm)
        new_row.append(row[-1])
        writer.writerow(new_row)

def searchExon_GTExExon(Gene, Pos, Ref, Alt, gtx_exon):
    Pos, LenV = int(Pos), len(Ref)-len(Alt)
    gene_obj = gtx_exon[gtx_exon["Gene"]==Gene]
    _Exons, Transcripts = [],[]
    for i, row in gene_obj.iterrows():
        start, end = row["start_pos"], row["end_pos"]
        if Pos > start -3 and Pos < end +3:
            return row["exon_id"]
        elif LenV > 0:
            if (Pos < start-3 and Pos + LenV > start ) or (Pos < end and Pos + LenV > end +3):
                return row["exon_id"]
    return "NA"

def searchExon_BrainSpan(Gene, Chr, Pos, Ref, Alt, ExonRow, ExonCol):
    Pos, LenV = int(Pos), len(Ref)-len(Alt)
    ExonRow = ExonRow[ExonRow["gene_symbol"]==Gene]
    ith = 1
    for i, row in ExonRow.iterrows():
        row_num, start, end = row["row_num"],int(row["start"]),int(row["end"])
        if Pos > start -3 and Pos < end + 3:
            if ith == ExonRow.shape[0]:
                return row_num
            else:
                return row_num
        elif LenV > 0: # a delition may span a splice site
            if (Pos < start-3 and Pos + LenV > start ) or (Pos < end and Pos + LenV > end +3):
                if ith == ExonRow.shape[0]:
                    return row_num
                else:
                    return row_num
        ith += 1
    return 0

def RelativeExonExp2GeneExp(GeneRPKM, SelectedTissueSamples, ExonRPKM, GTEx_LGDs, Genes, Tissue):
    ALL_EXON_REL, ALL_GENE_ZSCORE = [], []
    for EnsGeneID in Genes:
        Gene_LGDs = GTEx_LGDs[GTEx_LGDs["SEVERE_GENE"]==EnsGeneID]
        WithLGDSamples = Gene_LGDs["INDV"].values
        WithLGDDoners = set([x.split("-")[1] for x in WithLGDSamples])
        #TissueWithLGD = SelectedTissueSamples[(SelectedTissueSamples["Doner"].isin(WithLGDDoners)) &  (SelectedTissueSamples["SMTS"]==Tissue)]
        _GeneRPKM = GeneRPKM[GeneRPKM["Name"]==EnsGeneID] # RPKM of selected Gene
        if _GeneRPKM.shape[0] == 0:
            continue
        ExpDat = SelectedTissueSamples[SelectedTissueSamples["SMTS"]==Tissue] # GTEx meta data of selected Tissue
        Log2ExpOfGeneTissue = []
        for sp in ExpDat["SAMPID"].values:
            Log2ExpOfGeneTissue.append(_GeneRPKM[sp])
        Log2ExpOfGeneTissue = [math.log2(x+1) for x in Log2ExpOfGeneTissue]
        GeneTissueMean = np.mean(Log2ExpOfGeneTissue)
        GeneTissueStd = np.std(Log2ExpOfGeneTissue)
        VarinGene = GTEx_LGDs[GTEx_LGDs["SEVERE_GENE"]==EnsGeneID] # Mutation in selected Gene
        for i, row in VarinGene.iterrows():
            _ExonRPKM = ExonRPKM[ExonRPKM["Name"]==row["GTExExonID"]]
            if _ExonRPKM.shape[0] == 0:
                continue
            doner = row["INDV"].split("-")[1]
            samples = ExpDat[ExpDat["Doner"]==doner]["SAMPID"].values # all samples from the doner and selected Tissue
            if len(samples) == 0:
                continue
            gene_exps = []
            exon_exps = []
            for sp in samples:
                gene_exps.append(math.log2(_GeneRPKM[sp]+1))
                exon_exps.append(math.log2(_ExonRPKM[sp]+1))
            exon_rel_exp = np.mean(exon_exps) / np.mean(gene_exps)
            #exon_rel_exp = np.mean(exon_exps) / GeneTissueMean 
            gene_exp_zscore = (np.mean(gene_exps) - GeneTissueMean)/GeneTissueStd
            if exon_rel_exp == exon_rel_exp and gene_exp_zscore == gene_exp_zscore:
                ALL_EXON_REL.append(exon_rel_exp)
                ALL_GENE_ZSCORE.append(gene_exp_zscore)
    return ALL_EXON_REL, ALL_GENE_ZSCORE

##############################################################################
# AA distance; Frac.Trunc; Phenotypic Diff
##############################################################################

def LoadGTF(FileName):
    GTFTree = {}
    hand = open(FileName, 'rt')
    for l in hand:
        if l.startswith("#"):
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        CHR = llist[0].lstrip("chr")
        strand = llist[6]
        start = int(llist[3])
        end = int(llist[4])
        if llist[2] == "gene":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            GTFTree[gene_name] = GTFGene(gene_name, gene_id, strand)
        elif llist[2] == "transcript":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"]
            transcript_name = info["transcript_id"].split(".")[0]
            transcript_id = info["transcript_id"].split(".")[0]
            transcript_type = info["transcript_type"]
            if transcript_id not in GTFTree[gene_name].Transcripts and transcript_type=="protein_coding":
                GTFTree[gene_name].Transcripts[transcript_id] = GTFTranscript(gene_name, transcript_name, transcript_id, strand)
        elif llist[2] == "exon":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            exon_id = info["exon_id"].split(".")[0]
            exon_number = info["exon_number"]
            transcript_name = info["transcript_id"].split(".")[0]
            transcript_id = info["transcript_id"].split(".")[0]
            transcript_type = info["transcript_type"]
            if transcript_type=="protein_coding":
                exon= GTFExon(exon_id, start, end, transcript_id, strand)
                GTFTree[gene_name].Transcripts[transcript_id].Exons[exon_id] = exon
        elif llist[2] == "CDS":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            exon_id = info["exon_id"].split(".")[0]
            exon_number = info["exon_number"]
            transcript_name = info["transcript_id"].split(".")[0]
            transcript_id = info["transcript_id"].split(".")[0]
            transcript_type = info["transcript_type"]
            if transcript_type=="protein_coding":
                cds = GTFCDS(exon_id, start, end, transcript_id, strand)
                GTFTree[gene_name].Transcripts[transcript_id].Exons[exon_id].cds = cds
    return GTFTree


def annotatePP(row, location2pp):
    vcfVariant = row["vcfVariant"]
    Chr, Pos, Ref, Alt = vcfVariant.split(":")
    Pos = int(Pos)
    if "%s:%d"%(Chr, Pos) in location2pp:
        return location2pp["%s:%d"%(Chr, Pos)]
    elif "%s:%d"%(Chr, Pos+1) in location2pp:
        return location2pp["%s:%d"%(Chr, Pos+1)]
    elif "%s:%d"%(Chr, Pos-1) in location2pp:
        return location2pp["%s:%d"%(Chr, Pos-1)]
    else:
        print("%s:%d"%(Chr, Pos))
        return "NA"
    
def annotateTrans(row, location2trans):
    vcfVariant = row["vcfVariant"]
    Chr, Pos, Ref, Alt = vcfVariant.split(":")
    Pos = int(Pos)
    if "%s:%d"%(Chr, Pos) in location2trans:
        return location2trans["%s:%d"%(Chr, Pos)]
    elif "%s:%d"%(Chr, Pos+1) in location2trans:
        return location2trans["%s:%d"%(Chr, Pos+1)]
    elif "%s:%d"%(Chr, Pos-1) in location2trans:
        return location2trans["%s:%d"%(Chr, Pos-1)]
    else:
        print("%s:%d"%(Chr, Pos))
        return "NA"

def searchAAPos(Pos, transobj):
    strand = transobj.strand
    Exons = transobj.ExonSeq
    Total = 0
    res = "NA"
    if strand == "+":
        exon = Exons[0]
        cds = exon if exon.cds == None else exon.cds
        last_start = cds.start
        last_end = cds.end
        Total += (last_end-last_start)
        if Pos > last_start and Pos < last_end:
            res = Pos - last_start
        for exon in Exons[1:]:
            cds = exon if exon.cds == None else exon.cds
            if Pos > last_end and Pos < cds.start: #intron variant
                res = Total 
            elif Pos > cds.start and Pos < cds.end:
                res = Pos - cds.start + Total
            last_start, last_end = cds.start, cds.end
            Total += (last_end-last_start)
    else:
        exon = Exons[0]
        cds = exon if exon.cds == None else exon.cds
        last_start = cds.start
        last_end = cds.end
        Total += (last_end-last_start)
        if Pos > last_start and Pos < last_end:
            res = last_end - Pos
        for exon in Exons[1:]:
            cds = exon if exon.cds == None else exon.cds
            if Pos < last_start and Pos > cds.end: #intron variant
                res = Total
            elif Pos > cds.start and Pos < cds.end:
                res = cds.end - Pos + Total
            last_start, last_end = cds.start, cds.end
            Total += (last_end-last_start)
    return res, Total

class DistPhenotype:
    def __init__(self, dist, phenotype):
        self.dist = dist
        self.logdist = math.log10(dist+1)
        self.phenotype = phenotype

def get_smaller_P(obs, null):
    count = 0
    for i,v in enumerate(null):
        if obs >= v:
            count += 1
    return float(count)/len(null)

def GetPairsForAAdistPhenotype(RecGeneDF, phenotype="NVIQ"):
    SameExonAA, SameExonIQ, DiffExonAA, DiffExonIQ = [], [], [], []
    for gene in list(set(RecGeneDF["effectGene"])):
        df = RecGeneDF[RecGeneDF["effectGene"]==gene]
        for row1, row2 in itertools.combinations(df.iterrows(), r=2):
            row1, row2 = row1[1], row2[1]
            IQdiff = abs(row1[phenotype]-row2[phenotype])
            PPdiff = abs(int(row1["ProteinPos"])-int(row2["ProteinPos"]))
            if PPdiff == 0:
                print(gene, row1["Transcript"], row1["vcfVariant"], row2["vcfVariant"])
                continue
            if row1["ExonID"] == row2["ExonID"]:
                SameExonAA.append(PPdiff)
                SameExonIQ.append(IQdiff)
                #print(gene, PPdiff, IQdiff)
            else:
                DiffExonAA.append(PPdiff)
                DiffExonIQ.append(IQdiff)
    return SameExonAA, SameExonIQ, DiffExonAA, DiffExonIQ

def AccetRejectSamplingForSameDiffExonAAdist(SameExonAA, SameExonPheno, DiffExonAA, DiffExonPheno, M = 20, N = 1000, SampleSetIncludeSameExon=False):
    if SampleSetIncludeSameExon:
        D1 = [math.log10(x) for x in SameExonAA]
        D2 = [math.log10(x) for x in SameExonAA + DiffExonAA]
        pairs = list(zip(SameExonAA, SameExonPheno)) + list(zip(DiffExonAA, DiffExonPheno))
    else:
        D1 = [math.log10(x) for x in SameExonAA]
        D2 = [math.log10(x) for x in DiffExonAA]
        pairs = list(zip(DiffExonAA, DiffExonPheno))
    LogDist = []
    for dist, IQ in pairs:
        LogDist.append(DistPhenotype(dist, IQ))
    mu1 = np.mean(D1)
    std1 = np.std(D1)
    #print(mu1, std1)
    mu2 = np.mean(D2)
    std2 = np.std(D2)
    #print(mu2, std2)
    f = lambda x : scipy.stats.norm.pdf(x, loc=mu1, scale=std1)
    g = lambda x : scipy.stats.norm.pdf(x, loc=mu2, scale=std2)
    AVGPhenoDiffs = []
    AVG_dists = []
    for i in range(N):
        j = 0
        x_samples = []
        while j < len(SameExonPheno):
            u = np.random.uniform(0, 1)
            x_sample = np.random.choice(LogDist, size=1, replace=True)
            if u < f(x_sample[0].logdist) / (M * g(x_sample[0].logdist)):
                x_samples.append(x_sample)
                j += 1
        AVGPhenoDiffs.append(np.mean([dist[0].phenotype for dist in x_samples]))
        AVG_dists.append(np.mean([dist[0].logdist for dist in x_samples]))
        if i % 100 == 0:
            sys.stdout.write("\r{}".format(i))
    return AVGPhenoDiffs, AVG_dists


def GetPairsForPheno1Pheno2(RecGeneDF, phenotype1="NVIQ", phenotype2="VIQ"):
    SameExonAA, SameExonIQ, DiffExonAA, DiffExonIQ = [], [], [], []
    for gene in list(set(RecGeneDF["effectGene"])):
        df = RecGeneDF[RecGeneDF["effectGene"]==gene]
        for row1, row2 in itertools.combinations(df.iterrows(), r=2):
            row1, row2 = row1[1], row2[1]
            IQdiff = abs(row1[phenotype]-row2[phenotype])
            PPdiff = abs(int(row1["ProteinPos"])-int(row2["ProteinPos"]))
            if PPdiff == 0:
                print(gene, row1["Transcript"], row1["vcfVariant"], row2["vcfVariant"])
                continue
            if row1["ExonID"] == row2["ExonID"]:
                SameExonAA.append(PPdiff)
                SameExonIQ.append(IQdiff)
                #print(gene, PPdiff, IQdiff)
            else:
                DiffExonAA.append(PPdiff)
                DiffExonIQ.append(IQdiff)
    return SameExonAA, SameExonIQ, DiffExonAA, DiffExonIQ

def AccetRejectSamplingForPhenotypesRelationship(SameExonP1, SameExonP2, DiffExonP1, DiffExonP2, M = 20, N = 1000, SampleSetIncludeSameExon=False):
    if SampleSetIncludeSameExon:
        D1 = [x for x in SameExonP1]
        D2 = [x for x in SameExonP1 + DiffExonP1]
        pairs = list(zip(SameExonP1, SameExonP2)) + list(zip(DiffExonP1, DiffExonP2))
    else:
        D1 = [x for x in SameExonP1]
        D2 = [x for x in DiffExonP1]
        pairs = list(zip(DiffExonP1, DiffExonP2))
    LogDist = []
    for P1, P2 in pairs:
        LogDist.append(DistPhenotype(P1, P2))
    mu1 = np.mean(D1)
    std1 = np.std(D1)
    #print(mu1, std1)
    mu2 = np.mean(D2)
    std2 = np.std(D2)
    #print(mu2, std2)
    f = lambda x : scipy.stats.norm.pdf(x, loc=mu1, scale=std1)
    g = lambda x : scipy.stats.norm.pdf(x, loc=mu2, scale=std2)
    AVGPhenoDiffs = []
    AVG_dists = []
    for i in range(N):
        j = 0
        x_samples = []
        while j < len(SameExonP1):
            u = np.random.uniform(0, 1)
            x_sample = np.random.choice(LogDist, size=1, replace=True)
            if u < f(x_sample[0].dist) / (M * g(x_sample[0].dist)):
                x_samples.append(x_sample)
                j += 1
        AVGPhenoDiffs.append(np.mean([dist[0].phenotype for dist in x_samples]))
        AVG_dists.append(np.mean([dist[0].dist for dist in x_samples]))
        if i % 100 == 0:
            sys.stdout.write("\r{}".format(i))
    return AVGPhenoDiffs, AVG_dists

def SameExonDef1(DF):
    SameExon = []
    SameExonSameGender = []
    SameGene = []
    ALLPairs = []
    for row1, row2 in itertools.combinations(DF.iterrows(), r=2):
        row1,row2 = row1[1], row2[1]
        score1 = row1["composite_standard_score"]
        score2 = row2["composite_standard_score"]
        diff = abs(score1-score2)
        ALLPairs.append(diff)
        if row1["genetic_status"] == row2["genetic_status"]:
            SameGene.append(diff)
            if row1["Exons"] == row2["Exons"]:
                if row1["isLEJ"] == "T" or row2["isLEJ"] == "T":
                    continue
                SameExon.append(diff)
                if row1["sex"] == row2["sex"]:
                    SameExonSameGender.append(diff)
    return SameExon, SameExonSameGender, SameGene, ALLPairs

def SameExonDef2(DF):
    SameExon = []
    SameExonSameGender = []
    SameGene = []
    ALLPairs = []
    for i, row in DF.iterrows():
        sfari_id, sex, score, Exon, Gene = row["sfari_id"], row["sex"], row["composite_standard_score"], row["Exons"], row["genetic_status"]
        OtherSameExon = DF[(DF["Exons"]==Exon) & (DF["sfari_id"]!=sfari_id)]
        OtherSameGene = DF[(DF["genetic_status"]==Gene) & (DF["sfari_id"]!=sfari_id)]
        OtherSameExonSameGender = OtherSameExon[OtherSameExon["sex"]==sex] 
        ALLPairs.append(abs(score-np.mean(DF["composite_standard_score"].values)))
        if OtherSameGene.shape[0] >= 1:
            SameGene.append(abs(score-np.mean(OtherSameGene["composite_standard_score"].values)))
        if OtherSameExon.shape[0] >= 1:
            SameExon.append(abs(score-np.mean(OtherSameExon["composite_standard_score"].values)))
        if OtherSameExonSameGender.shape[0] >= 1:
            SameExonSameGender.append(abs(score-np.mean(OtherSameExonSameGender["composite_standard_score"].values)))
    return SameExon, SameExonSameGender, SameGene, ALLPairs

def PairSort(X, Y):
    XY = zip(X, Y)
    sorted_XY = sorted(XY, key=lambda x:x[0])
    X, Y = [],[]
    for x,y in sorted_XY:
        if x == x and y==y:
            X.append(x)
            Y.append(y)
    return XY, X, Y

"""
def MyMovingAVG(X, Y, StepSize, Overlap=0.5):
    minX, maxX = min(X), max(X)
    XY = zip(X, Y)
    sorted_XY = sorted(XY)
    #print(sorted_XY)
    Xs, Ys = [], []
    last_start = minX
    last_end = last_start+Overlap*StepSize
    last_mid = (last_start + last_end)/2
    last_idx = 0
    while 1:
        start, end = last_mid, last_mid+StepSize
        mid = (start + end)/2
        if start > maxX:
            break
        while 1:
            if sorted_XY[j][0] > mid:
                tmp_idx = j
            if sorted_XY[j][0] > end:
                break
            _XY.append(sorted_XY[j])
            j += 1
        last_idx = tmp_idx
        last_mid 
        Xs.append(mid)
        Ys.append(np.mean([Y for X,Y in _XY]))
    return Xs, Ys
    for i in range(Nwindow):
        _XY = sorted_XY[i*Nwindow:(i+1)*Nwindow]
        _X = i * step
        _Y = np.mean([Y for X,Y in _XY])
        Xs.append(_X)
        Ys.append(_Y)
    return Xs, Ys
"""

def MyMovingAVG(X, Y, StepSize, overlap=0):
    XY, X, Y = PairSort(X,Y)
    step = 0
    res_X, res_Y = [], []
    while 1:
        if step > len(X):
            break
        X_ = X[step:step+StepSize]
        Y_ = Y[step:step+StepSize]
        res_X.append(np.mean(X_))
        res_Y.append(np.mean(Y_))
        step += StepSize
    return res_X, res_Y

def SSC_LGD_2_VCF(DF):
    def vcfVariant(row):
        return row["vcfVariant"].split(":")
    DF["Chr"] = DF.apply(lambda row:vcfVariant(row)[0], axis=1)
    DF["Pos"] = DF.apply(lambda row:vcfVariant(row)[1], axis=1)
    DF["RSID"] = DF["KEY"]
    DF["Ref"] = DF.apply(lambda row:vcfVariant(row)[2], axis=1)
    DF["Alt"] = DF.apply(lambda row:vcfVariant(row)[3], axis=1)
    DF = DF[["Chr", "Pos", "RSID", "Ref", "Alt"]]
    return DF


##############################################################################
# Burden of Developmental stages
##############################################################################
def SubSetBrainSpanData(GeneSet, ID="ensembl"):
    exon_row_meta = csv.reader(open("/Users/jiayao/Work/BrainDisorders/data/expression/brainspan/exons_matrix/rows_metadata.csv", 'rt'))
    exon_exp = csv.reader(open("/Users/jiayao/Work/BrainDisorders/data/expression/brainspan/exons_matrix/qn_exons_matrix.csv", 'rt'))

    subset_exon_row_mata = csv.writer(open("./brainspan/rows_metadata.csv", 'wt'))
    subset_exon_exp = csv.writer(open("./brainspan/ssc.lgd.qn.exons_matrix.csv",'wt'))

    exon_row_meta_head = next(exon_row_meta)
    subset_exon_row_mata.writerow(exon_row_meta_head) 
    for meta_row, exp_row in zip(exon_row_meta, exon_exp):
        XX = dict(zip(exon_row_meta_head, meta_row))
        #if 
        if XX["ensembl_gene_id"] in GeneSet:
        #if XX["gene_symbol"] in GeneSet:
            subset_exon_row_mata.writerow(meta_row)
            subset_exon_exp.writerow(exp_row)

def SubSetBrainSpanDataGene(GeneSet, ID="ensembl"):
    exon_row_meta = csv.reader(open("/Users/jiayao/Work/BrainDisorders/data/expression/brainspan/gene_matrix/rows_metadata.csv", 'rt'))
    exon_exp = csv.reader(open("/Users/jiayao/Work/BrainDisorders/data/expression/brainspan/gene_matrix/qn.expression_matrix.csv", 'rt'))

    subset_exon_row_mata = csv.writer(open("./brainspan/gene_rows_metadata.csv", 'wt'))
    subset_exon_exp = csv.writer(open("./brainspan/ssc.lgd.qn.gene_matrix.csv",'wt'))

    exon_row_meta_head = next(exon_row_meta)
    subset_exon_row_mata.writerow(exon_row_meta_head) 
    for meta_row, exp_row in zip(exon_row_meta, exon_exp):
        XX = dict(zip(exon_row_meta_head, meta_row))
        #if 
        if XX["ensembl_gene_id"] in GeneSet:
        #if XX["gene_symbol"] in GeneSet:
            subset_exon_row_mata.writerow(meta_row)
            subset_exon_exp.writerow(exp_row)

def MakeExonID(exon_row_meta):
    #ALL_GENE = list(set(exon_row_meta["gene_symbol"]))
    #for GENE in ALL_GENE:
    #    df = exon_row_meta[exon_row_meta["gene_symbol"]]
    LAST_GENE = None
    for i, row in exon_row_meta.iterrows():
        #GENE = row["gene_symbol"]
        GENE = row["GENE.SYMBOL"]
        if GENE != LAST_GENE:
            idx = 1
            LAST_GENE = GENE
        else:
            idx += 1
        #EXONID = "_".join([GENE, str(idx)])
        EXONID = "{}_{}".format(GENE, idx)
        exon_row_meta.loc[i, "EXONID2"] = EXONID
        if i % 100 == 0:
           sys.stdout.write("\r{}".format(i)) 
    return exon_row_meta 


def GetExonExp(gene, ExonID, expdict, ins):
    # expdict[gene][ExonID]
    tmp = []
    for stage in ins.Stages:
        #tmp.append(np.mean([math.log2(1+x) for x in expdict[gene][ExonID][stage]]))
        tmp.append(np.mean(expdict[gene][ExonID][stage]))
    return tmp


def GetGeneExpAndExonExp(ins):
    gene_exp_dict = {}
    for gene, exps in expdict_gene.items():
        tmp = []
        for stage in ins.Stages:
            #tmp.append(np.mean([math.log2(1+x) for x in exps[stage]]))
            tmp.append(np.mean(exps[stage]))
        gene_exp_dict[gene] = tmp
        #ExonID2Nmut = {}
        #for i, row in bp_exon_row_meta_with_gene.iterrows():
        #    exonid = row["row_num"]
        #    Vars = row["Vars"]
        #    if Vars != "":
        #        N = len(Vars.split(";"))
        #    else:
        #        N = 0
        #    ExonID2Nmut[exonid] = N

class EXON:
    def __init__(self, ID, ID2, Nmut, geneExp, exonExp, exonCDS, exonCDS2, exonCDS3, IQgt70=False, IQlt70=False, biasmethod="-"):
        self.ID = ID
        self.ID2 = ID2
        self.Nmut = Nmut
        #self.geneExp = geneExp
        self.exonExp = exonExp
        self.CDSLength = exonCDS
        self.CDSLength2 = exonCDS2
        self.CDSLength3 = exonCDS3
        self.ralExp = []
        self.IQgt70 = IQgt70
        self.IQlt70 = IQlt70
        if biasmethod == "/":
            if np.mean(self.exonExp[6:]) == 0:
                self.relbias = 0
            else:
                self.relbias = np.mean(self.exonExp[:6]) / np.mean(self.exonExp[6:])
        elif biasmethod == "-":
            self.relbias = np.mean(self.exonExp[:6]) - np.mean(self.exonExp[6:])
        if self.relbias < -10:
            self.relbias = -10
        if self.relbias > 10:
            self.relbias = 10


class EXON_mwu_bias:
    def __init__(self, ID, ID2, Nmut, geneExp, exonExp, exonExp_tmp, exonCDS, IQgt70=False, IQlt70=False, biasmethod="-"):
        self.ID = ID
        self.ID2 = ID2
        self.Nmut = Nmut
        self.exonExp = exonExp # Prenatal/Postnatal: 0-236
        self.exonExp_tmp = exonExp_tmp
        self.CDSLength = exonCDS
        self.IQgt70 = IQgt70
        self.IQlt70 = IQlt70
        self.prenatal_exps = self.exonExp[:237]
        self.postnatal_exps = self.exonExp[237:]
        if np.mean(self.postnatal_exps) != 0 :
            t, p = scipy.stats.mannwhitneyu(self.prenatal_exps, self.postnatal_exps)
            self.pvalue = p
        else:
            self.pvalue = 1
        if biasmethod == "/":
            if np.mean(self.exonExp[237:]) == 0:
                self.relbias = 0
            else:
                self.relbias = np.mean(self.exonExp[:237]) / np.mean(self.exonExp[237:])
        elif biasmethod == "-":
            self.relbias = np.mean(self.exonExp[:237]) - np.mean(self.exonExp[237:])
        if self.relbias < -10:
            self.relbias = -10
        if self.relbias > 10:
            self.relbias = 10

def PoolTheExons(ins, expdict_gene, expdict_exon, VarFile, Genes, bp_exon_row_meta_with_gene, ExonID2Length, minLog2RPKMplus1Cut = 0):
    ExonPool = []
    TargetedExonSet = set(VarFile["ExonID"].values)
    IQgt70ExonSet = set(VarFile[VarFile["NVIQ"]> 70]["ExonID"].values)
    IQlt70ExonSet = set(VarFile[VarFile["NVIQ"]<=70]["ExonID"].values)
    ExonCount = VarFile.groupby("ExonID")["ExonID"].count()
    #for gene in list(set(VarFile["effectGene"].values)):
    for gene in Genes:
        df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["symbol"]==gene]
        #for exonID in expdict_exon[gene].keys(): 
        for i, row in df.iterrows():
            geneExp = 0
            exonExp = GetExonExp(gene, exonID, expdict_exon, ins)
            exonCDS = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["cds length"].values[0]
            exonID2 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["EXONID2"].values[0]
            exonCDS2 = ExonID2Length[exonID2]
            #exonCDS3 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["cds_length_3"].values[0]
            exonCDS3 = exonCDS
            if exonCDS3 != exonCDS3:
                exonCDS3 = exonCDS
            if exonID in TargetedExonSet:
                Exon = EXON(exonID, exonID2, ExonCount[exonID], geneExp, exonExp, exonCDS, exonCDS2, exonCDS3)
                if exonID in IQgt70ExonSet:
                    Exon.IQgt70 = True
                if exonID in IQlt70ExonSet:
                    Exon.IQlt70 = True
            else:
                Exon = EXON(exonID, exonID2, 0, geneExp, exonExp, exonCDS, exonCDS2, exonCDS3)
            if np.mean(Exon.exonExp) > minLog2RPKMplus1Cut:
                ExonPool.append(Exon)
    return ExonPool


def MatchCDS3(JonCDS, exon_row_meta):
    #JonCDS_GENEs = set(JonCDS["ensembl_gene_id"].values)
    GeneSet = set(exon_row_meta["ensembl_gene_id"].values)
    for gene in list(GeneSet):
        GENESUBSET = exon_row_meta[exon_row_meta["ensembl_gene_id"]==gene]
        CDSSUBBET = JonCDS[JonCDS["ensembl_gene_id"]==gene]
        for i, row in GENESUBSET.iterrows():
            start, end = row["start"], row["end"]
            for j, row2 in CDSSUBBET.iterrows():
                cds_start, cds_end = row2["genomic_coding_start"], row2["genomic_coding_end"]
                if cds_start >= start and cds_end <= end: 
                    cds_len = row2["cds_end"] - row2["cds_start"] + 1
                    exon_row_meta.loc[i, "cds_length_3"] = cds_len 
                    break
    return exon_row_meta

def GetExonTimeExp(EXON_EXP_RPKM, Stage2Idx, stat="mean", log2=False):
    ExonID2EXP = {}
    for i, row in EXON_EXP_RPKM.iterrows():
        exonid = row[0]
        exps = []
        for stage in Stages:
            exp = []
            for idx in Stage2Idx[stage]:
                if log2:
                    exp.append(math.log2(row[idx]+1))
                else:
                    exp.append(row[idx])
            if stat == "median":
                exps.append(np.median(exp))
            elif stat == "mean":
                exps.append(np.mean(exp))
        ExonID2EXP[exonid] = exps
    return ExonID2EXP

def GetExonTimeExp_mwu_bias(EXON_EXP_RPKM, log2=False):
    ExonID2EXP = {}
    for i, row in EXON_EXP_RPKM.iterrows():
        exonid = row[0]
        exps = list(row[1:])
        if log2:
            exps = [math.log2(1+x) for x in exps]
        ExonID2EXP[exonid] = exps
    return ExonID2EXP

def GetGeneTimeExp(EXP_RPKM, Stage2Idx, stat="mean", log2=False):
    ID2EXP = {}
    for i, row in EXP_RPKM.iterrows():
        geneid = row[0]
        exps = []
        for stage in Stages:
            exp = []
            for idx in Stage2Idx[stage]:
                if log2:
                    exp.append(math.log2(row[idx]+1))
                else:
                    exp.append(row[idx])
            if stat == "median":
                exps.append(np.median(exp))
            elif stat == "mean":
                exps.append(np.mean(exp))
        ID2EXP[geneid] = exps
    return ID2EXP

def GetGeneTimeExp_mwu_bias(EXP_RPKM, log2=False):
    ID2EXP = {}
    for i, row in EXP_RPKM.iterrows():
        geneid = row[0]
        exps = list(row[1:])
        if log2:
            exps = [math.log2(1+x) for x in exps]
        ID2EXP[geneid] = exps
    return ID2EXP

def PoolTheExons2(ins, expdict_exon, VarFile, Genes, bp_exon_row_meta_with_gene, ExonID2Length, minLog2RPKMplus1Cut = 0, biasmethod="-"):
    ExonPool = []
    TargetedExonSet = set(VarFile["ExonID"].values)
    IQgt70ExonSet = set(VarFile[VarFile["NVIQ"]> 70]["ExonID"].values)
    IQlt70ExonSet = set(VarFile[VarFile["NVIQ"]<=70]["ExonID"].values)
    ExonCount = VarFile.groupby("ExonID")["ExonID"].count()
    for gene in Genes:
        geneExp = 0
        #df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["gene_symbol"]==gene]
        df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["ensembl_gene_id"]==gene]
        for i, row in df.iterrows():
            exonID = row["row_num"]
            exonExp = expdict_exon[row["row_num"]]
            exonCDS = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["cds length"].values[0]
            exonID2 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["EXONID2"].values[0]
            exonCDS2 = ExonID2Length[exonID2]
            #exonCDS3 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["cds_length_3"].values[0]
            exonCDS = exonCDS2
            exonCDS3 = exonCDS2
            #if exonCDS3 != exonCDS3:
            #    exonCDS3 = exonCDS
            if exonID in TargetedExonSet:
                if biasmethod == "-":
                    Exon = EXON(exonID, exonID2, ExonCount[exonID], geneExp, exonExp, exonCDS, exonCDS2, exonCDS3, biasmethod="-")
                elif biasmethod == "/":
                    Exon = EXON(exonID, exonID2, ExonCount[exonID], geneExp, exonExp, exonCDS, exonCDS2, exonCDS3, biasmethod="/")
                if exonID in IQgt70ExonSet:
                    Exon.IQgt70 = True
                if exonID in IQlt70ExonSet:
                    Exon.IQlt70 = True
            else:
                if biasmethod == "-":
                    Exon = EXON(exonID, exonID2, 0, geneExp, exonExp, exonCDS, exonCDS2, exonCDS3, biasmethod="-")
                elif biasmethod == "/":
                    Exon = EXON(exonID, exonID2, 0, geneExp, exonExp, exonCDS, exonCDS2, exonCDS3, biasmethod="/")
            #print(Exon.exonExp)
            if np.mean(Exon.exonExp) >= minLog2RPKMplus1Cut:
                ExonPool.append(Exon)
    return ExonPool

# expdict_exon : all 524 values
def PoolTheExons2_mnw_bias(ins, expdict_exon, expdict_temporal_exon, VarFile, Genes, bp_exon_row_meta_with_gene, ExonID2Length, minLog2RPKMplus1Cut = 0, biasmethod="-"):
    ExonPool = []
    TargetedExonSet = set(VarFile["ExonID"].values)
    IQgt70ExonSet = set(VarFile[VarFile["NVIQ"]> 70]["ExonID"].values)
    IQlt70ExonSet = set(VarFile[VarFile["NVIQ"]<=70]["ExonID"].values)
    ExonCount = VarFile.groupby("ExonID")["ExonID"].count()
    for gene in Genes:
        geneExp = 0
        #df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["gene_symbol"]==gene]
        df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["ensembl_gene_id"]==gene]
        for i, row in df.iterrows():
            exonID = row["row_num"]
            exonExp = expdict_exon[row["row_num"]]
            exonExp_tmp = expdict_temporal_exon[row["row_num"]]
            exonCDS = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["cds length"].values[0]
            exonID2 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["EXONID2"].values[0]
            exonCDS2 = ExonID2Length[exonID2]
            #exonCDS3 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["row_num"]==exonID]["cds_length_3"].values[0]
            exonCDS = exonCDS2
            if exonID in TargetedExonSet:
                if biasmethod == "-":
                    Exon = EXON_mwu_bias(exonID, exonID2, ExonCount[exonID], geneExp, exonExp, exonExp_tmp, exonCDS, biasmethod="-")
                elif biasmethod == "/":
                    Exon = EXON_mwu_bias(exonID, exonID2, ExonCount[exonID], geneExp, exonExp, exonExp_tmp, exonCDS, biasmethod="/")
                if exonID in IQgt70ExonSet:
                    Exon.IQgt70 = True
                if exonID in IQlt70ExonSet:
                    Exon.IQlt70 = True
            else:
                if biasmethod == "-":
                    Exon = EXON_mwu_bias(exonID, exonID2, 0, geneExp, exonExp, exonExp_tmp, exonCDS, biasmethod="-")
                elif biasmethod == "/":
                    Exon = EXON_mwu_bias(exonID, exonID2, 0, geneExp, exonExp, exonExp_tmp, exonCDS, biasmethod="/")
            #print(Exon.exonExp)
            if np.mean(Exon.exonExp) >= minLog2RPKMplus1Cut:
                ExonPool.append(Exon)
    return ExonPool

class GENE_mwu_bias:
    def __init__(self, ROWNUM, ENSID, SYMBOL, Nmut, geneExp, geneExp_tmp, geneCDS, IQgt70=False, IQlt70=False, biasmethod="-"):
        self.id = ROWNUM
        self.ensid = ENSID
        self.symbol = SYMBOL
        self.Nmut = Nmut
        self.geneExp = geneExp # Prenatal/Postnatal: 0-236
        self.geneExp_tmp = geneExp_tmp
        self.CDSLength = geneCDS
        self.IQgt70 = IQgt70
        self.IQlt70 = IQlt70
        self.prenatal_exps = self.geneExp[:237]
        self.postnatal_exps = self.geneExp[237:]
        if np.mean(self.postnatal_exps) != 0 :
            #t, p = scipy.stats.mannwhitneyu(self.prenatal_exps, self.postnatal_exps, alternative="two-sided")
            t, p = scipy.stats.mannwhitneyu(self.prenatal_exps, self.postnatal_exps)
            self.pvalue = p
        else:
            self.pvalue = 1
        if biasmethod == "/":
            if np.mean(self.geneExp[237:]) == 0:
                self.relbias = 0
            else:
                self.relbias = np.mean(self.geneExp[:237]) / np.mean(self.geneExp[237:])
        elif biasmethod == "-":
            self.relbias = np.mean(self.geneExp[:237]) - np.mean(self.geneExp[237:])
        if self.relbias < -10:
            self.relbias = -10
        if self.relbias > 10:
            self.relbias = 10

def GetGeneCDS(bp_exon_row_meta, bp_gene_row_meta, exonid2cds):
    GeneID2CDS = {}
    for i, row in bp_gene_row_meta.iterrows():
        gene = row["ensembl_gene_id"]
        #df = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==gene]
        df = bp_exon_row_meta[bp_exon_row_meta["ensembl_gene_id"]==gene]
        CDS = 0
        for i, row in df.iterrows():
            CDS += exonid2cds[row["EXONID2"]]
        GeneID2CDS[gene] = CDS
    return GeneID2CDS
def GetGeneCDS2(bp_exon_row_meta, Genes, exonid2cds):
    GeneID2CDS = {}
    for gene in Genes:
        #df = bp_exon_row_meta[bp_exon_row_meta["gene_symbol"]==gene]
        df = bp_exon_row_meta[bp_exon_row_meta["ensembl_gene_id"]==gene]
        CDS = 0
        for i, row in df.iterrows():
            CDS += exonid2cds[row["EXONID2"]]
        GeneID2CDS[gene] = CDS
    return GeneID2CDS

def PoolTheGenes2_mnw_bias(ins, expdict_gene, expdict_temporal_gene, VarFile, Genes, bp_gene_row_meta, GeneID2Length, minLog2RPKMplus1Cut = 0, biasmethod="-"):
    GenePool = []
    TargetedGeneSet = set(VarFile["ENSGID"].values)
    IQgt70GeneSet = set(VarFile[VarFile["NVIQ"]> 70]["ENSGID"].values)
    IQlt70GeneSet = set(VarFile[VarFile["NVIQ"]<=70]["ENSGID"].values)
    df1 = bp_gene_row_meta[bp_gene_row_meta["ensembl_gene_id"].isin(TargetedGeneSet)]
    df2 = bp_gene_row_meta[~bp_gene_row_meta["ensembl_gene_id"].isin(TargetedGeneSet)]
    #print(df.shape)
    #GeneCount = VarFile.groupby("effectGene")["effectGene"].count()
    GeneCount = VarFile.groupby("ENSGID")["ENSGID"].count()
    for i, row in df1.iterrows():
        ENSID = row["ensembl_gene_id"]
        geneRowNum = row["row_num"] 
        geneSymbol = row["gene_symbol"] 
        #if geneSymbol == "MLL5":
        #    geneSymbol = "KMT2E"
        geneExp = expdict_gene[geneRowNum]
        geneExp_tmp = expdict_temporal_gene[geneRowNum]
        geneCDS = GeneID2Length[ENSID]
        if biasmethod == "-":
            Gene = GENE_mwu_bias(geneRowNum, ENSID, geneSymbol, GeneCount[ENSID], geneExp, geneExp_tmp, geneCDS, biasmethod="-")
        elif biasmethod == "/":
            Gene = GENE_mwu_bias(geneRowNum, ENSID, geneSymbol, GeneCount[ENSID], geneExp, geneExp_tmp, geneCDS, biasmethod="/")
        if ENSID in IQgt70GeneSet:
            Gene.IQgt70 = True
        if ENSID in IQlt70GeneSet:
            Gene.IQlt70 = True
        if np.mean(Gene.geneExp) >= minLog2RPKMplus1Cut:
            GenePool.append(Gene)
    for i, row in df2.iterrows():
        ENSID = row["ensembl_gene_id"]
        geneRowNum = row["row_num"] 
        geneSymbol = row["gene_symbol"] 
        #if geneSymbol == "MLL5":
        #    geneSymbol = "KMT2E"
        geneExp = expdict_gene[geneRowNum]
        geneExp_tmp = expdict_temporal_gene[geneRowNum]
        geneCDS = GeneID2Length[ENSID]
        if biasmethod == "-":
            Gene = GENE_mwu_bias(geneRowNum, ENSID, geneSymbol, 0, geneExp, geneExp_tmp, geneCDS, biasmethod="-")
        elif biasmethod == "/":
            Gene = GENE_mwu_bias(geneRowNum, ENSID, geneSymbol, 0, geneExp, geneExp_tmp, geneCDS, biasmethod="/")
        if ENSID in IQgt70GeneSet:
            Gene.IQgt70 = True
        if ENSID in IQlt70GeneSet:
            Gene.IQlt70 = True
        if np.mean(Gene.geneExp) >= minLog2RPKMplus1Cut:
            GenePool.append(Gene)
    return GenePool

class EXON2:
    def __init__(self, ID, ID2, Nmut, exonExp, exonCDS, IQgt70=False, IQlt70=False):
        self.ID = ID
        self.ID2 = ID2
        self.Nmut = Nmut
        self.exonExp = exonExp
        self.CDSLength = exonCDS
        self.IQgt70 = IQgt70
        self.IQlt70 = IQlt70
        self.relbias = np.mean(self.exonExp[:6]) - np.mean(self.exonExp[6:])
        if self.relbias < -10:
            self.relbias = -10
        if self.relbias > 10:
            self.relbias = 10

def PoolTheExons3(ins, VarFile, Genes, bp_exon_row_meta_with_gene, ExonID2Length, minLog2RPKMplus1Cut = 0):
    ExonPool = []
    TargetedExonSet = set(VarFile["ExonID"].values)
    IQgt70ExonSet = set(VarFile[VarFile["NVIQ"]> 70]["ExonID"].values)
    IQlt70ExonSet = set(VarFile[VarFile["NVIQ"]<=70]["ExonID"].values)
    ExonCount = VarFile.groupby("ExonID")["ExonID"].count()
    for gene in Genes:
        geneExp = 0
        #df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["GENE.SYMBOL"]==gene]
        df = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["gene_symbol"]==gene]
        for i, row in df.iterrows():
            exonID = row["ROW"]
            #exonExp = expdict_exon[row["ROW"]]
            exonExp = []
            for xx in range(1,13):
                exonExp.append(row["PERIOD.%d"%xx])
            exonID2 = bp_exon_row_meta_with_gene[bp_exon_row_meta_with_gene["ROW"]==exonID]["EXONID2"].values[0]
            exonCDS = ExonID2Length[exonID2]
            if exonID in TargetedExonSet:
                Exon = EXON2(exonID, exonID2, ExonCount[exonID], exonExp, exonCDS)
                if exonID in IQgt70ExonSet:
                    Exon.IQgt70 = True
                if exonID in IQlt70ExonSet:
                    Exon.IQlt70 = True
            else:
                Exon = EXON2(exonID, exonID2, 0, exonExp, exonCDS)
            if np.mean(Exon.exonExp) >= minLog2RPKMplus1Cut:
                ExonPool.append(Exon)
    return ExonPool

def LoadHighConfVarFil():
    VarFile = pd.read_excel("/Users/jiayao/Work/BrainDisorders/data/DenovoVariants/wigler2014LGD.xlsx")
    ASD_high_conf_df_ = [x.strip() for x in open(ASD_high_conf, 'rt')]
    VarFile = VarFile[VarFile["effectGene"].isin(ASD_high_conf_df_)]
    VarFile.loc[VarFile["effectGene"]=="KMT2E", "effectGene"] = "MLL5" #MLL5
    VarFile.loc[VarFile["effectGene"]=="SKIDA1", "effectGene"] = "C10orf140" #MLL5
    VarFile.loc[VarFile["effectGene"]=="KMT2A", "effectGene"] = "MLL" #MLL5 #
    VarFile.loc[VarFile["effectGene"]=="KMT2C", "effectGene"] = "MLL3" #MLL5
    wigler_fam_info = pd.read_excel("/Users/jiayao/Work/BrainDisorders/data/nature13908-s2/Supplementary_Table_1.xlsx")
    famID2Gender = dict(zip(wigler_fam_info["familyId"].values, wigler_fam_info["probandGender"].values))
    famID2VIQ = dict(zip(wigler_fam_info["familyId"].values, wigler_fam_info["probandVIQ"].values))
    famID2NVIQ = dict(zip(wigler_fam_info["familyId"].values, wigler_fam_info["probandNVIQ"].values))
    for i, row in VarFile.iterrows():
        famid, gene, (Chr, Pos, Ref, Alt) = row["familyId"], row["effectGene"], row["vcfVariant"].split(":")
        exonId = searchExon_BrainSpan(gene, Chr, Pos, Ref, Alt, bp_exon_row_meta, bp_exon_col_meta)
        VarFile.loc[i, "ExonID"] = exonId
        VarFile.loc[i, "Gender"] = famID2Gender[famid]
        VarFile.loc[i, "VIQ"] = famID2VIQ[famid]
        VarFile.loc[i, "NVIQ"] = famID2NVIQ[famid]


def BinExon(ExonPool, Nbins=4):
    #ExonPool.sort(key=lambda x: x.relbias)
    parts = []
    step = int(len(ExonPool)/Nbins)
    #print (step)
    ExonPool = sorted(ExonPool, key=lambda x: x.relbias, reverse=True)
    for i in range(Nbins-1):
        parts.append(ExonPool[i*step:(i+1)*step])
    parts.append(ExonPool[(Nbins-1)*step:])
    return parts


def BinExon_mwu_bias(ExonPool, cut = 1, Nbins=3, PvalueCut=0.05):
    #ExonPool.sort(key=lambda x: x.relbias)
    parts = [[],[],[]]
    step = int(len(ExonPool)/Nbins)
    #print (step)
    #ExonPool = sorted(ExonPool, key=lambda x: x.pvalue, reverse=False)
    for exon in ExonPool:
        if exon.pvalue < PvalueCut and exon.relbias > cut:
            parts[0].append(exon)
        elif exon.pvalue < PvalueCut and exon.relbias < cut:
            parts[2].append(exon)
        else:
            parts[1].append(exon)
    return parts

def BinExon_UseAndyBin(ExonPool, ExonID2Part):
    parts = [[],[],[],[]]
    for exon in ExonPool:
        if ExonID2Part[exon.ID2] == 'd':
            parts[0].append(exon)
        elif ExonID2Part[exon.ID2] == 'c':
            parts[1].append(exon)
        elif ExonID2Part[exon.ID2] == 'b':
            parts[2].append(exon)
        elif ExonID2Part[exon.ID2] == 'a':
            parts[3].append(exon)
    return parts

def PlotExonBins(bins, Nbins=4, gene=False):
    #labels = ["Strong prenatal bias","Prenatal bias","Weak bias","postnatal bias"]
    labels = ["1","2","3","4"]
    for i in range(Nbins):
        allexonexp = []
        for exon in bins[i]:
            if gene:
                allexonexp.extend(exon.geneExp_tmp)
            else:
                allexonexp.extend(exon.exonExp_tmp)
        Mean = np.mean(allexonexp)
        AvgExps, Errbars = [],[]
        for j,stage in enumerate(Stages):
            if gene:
                values = [x.geneExp_tmp[j] for x in bins[i]]
            else:
                values = [x.exonExp_tmp[j] for x in bins[i]]
            AvgExp = math.log2(np.mean(values)/Mean)
            Errbar = np.std([math.log2(x/Mean+1) for x in values])/math.sqrt(len(values))
            AvgExps.append(AvgExp)
            Errbars.append(Errbar)
        plt.errorbar(range(2,14), AvgExps, Errbars, label=labels[i])
    plt.legend()
    plt.show()

def PlotExonHist(bins):
    color=["blue", "orange","green","red"]
    for i in range(4):
        bias=[]
        for exon in bins[i]:
            #plt.plot(range(2,14), exon.exonExp)
            bias.append(exon.relbias)
        plt.hist(bias, bins=30, color=color[i], alpha=0.5, density=True)
    plt.show()

def GeneBinsBurden(parts, Nbins=3, Null="All", NormRate=False, NHighIQProband = 1870, NLowIQProband = 638, title=""):
    Nmut_ALL, Nmut_HighIQ, Nmut_LowIQ = [],[],[]
    CDS_ALL, CDS_HighIQ, CDS_LowIQ = [],[],[]
    Rate_ALL, Rate_HighIQ, Rate_LowIQ = [],[],[]
    #TotalALLRate, TotalHighIQRate, TotalLowIQRate = 0,0,0
    #TotalMuts_ALL, TotalMuts_HighIQ, TotalMuts_LowIQ = 0,0,0
    #TotalCDS_ALL,TotalCDS_HighIQ,TotalCDS_LowIQ = 0,0,0
    for i in range(Nbins):
        Nmut_ALL.append(sum([x.Nmut for x in parts[i]]))
        Nmut_HighIQ.append(sum([x.Nmut for x in parts[i] if x.IQgt70]))
        Nmut_LowIQ.append(sum([x.Nmut for x in parts[i] if x.IQlt70]))
        CDS_ALL.append(sum([x.CDSLength for x in parts[i]]))
        #CDS_HighIQ.append(sum([x.CDSLength for x in parts[i] if x.IQgt70]))
        #CDS_LowIQ.append(sum([x.CDSLength for x in parts[i] if x.IQlt70]))
        CDS_HighIQ.append(sum([x.CDSLength for x in parts[i]]))
        CDS_LowIQ.append(sum([x.CDSLength for x in parts[i]]))
        Rate_ALL.append(Nmut_ALL[i]/CDS_ALL[i])
        Rate_HighIQ.append(Nmut_HighIQ[i]/CDS_HighIQ[i])
        Rate_LowIQ.append(Nmut_LowIQ[i]/CDS_LowIQ[i])
    TotalMutRate_ALL = sum(Nmut_ALL)/sum(CDS_ALL)
    TotalMutRate_HighIQ = sum(Nmut_HighIQ)/sum(CDS_HighIQ)
    TotalMutRate_LowIQ = sum(Nmut_LowIQ)/sum(CDS_LowIQ)
    if Null == "All":
        p1 = scipy.stats.binom_test(Nmut_ALL[0], sum(Nmut_ALL), p=CDS_ALL[0]/sum(CDS_ALL)) # All as null
        p2 = scipy.stats.binom_test(Nmut_ALL[2], sum(Nmut_ALL), p=CDS_ALL[2]/sum(CDS_ALL)) # All as null
    elif Null == "NS":
        p1 = scipy.stats.binom_test(Nmut_ALL[0], Nmut_ALL[0]+Nmut_ALL[1], p=CDS_ALL[0]/(CDS_ALL[0]+CDS_ALL[1])) # NS as null
        p2 = scipy.stats.binom_test(Nmut_ALL[2], Nmut_ALL[2]+Nmut_ALL[1], p=CDS_ALL[2]/(CDS_ALL[2]+CDS_ALL[1])) # NS as null
    print("All proband binom P:", p1, p2)
    normedRate0 = []
    for Rate, group in zip(Rate_ALL, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        if NormRate:
            xx = Rate/TotalMutRate_ALL
        else:
            xx = Rate/(NHighIQProband+NLowIQProband)
        normedRate0.append(xx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2), dpi=80)
    fig.suptitle(title)
    ax1.plot(normedRate0, color="black", label="all proband")
    #ax1.axhline(y=1, color="black", linestyle=":")
    ax1.legend()
    normedRate1 = []
    normedRate2 = []
    if Null == "All":
        p1 = scipy.stats.binom_test(Nmut_HighIQ[0], sum(Nmut_HighIQ), p=CDS_HighIQ[0]/sum(CDS_HighIQ))
        p2 = scipy.stats.binom_test(Nmut_HighIQ[2], sum(Nmut_HighIQ), p=CDS_HighIQ[2]/sum(CDS_HighIQ))
    elif Null == "NS":
        p1 = scipy.stats.binom_test(Nmut_HighIQ[0], Nmut_HighIQ[0]+Nmut_HighIQ[1], p=CDS_HighIQ[0]/(CDS_HighIQ[0]+CDS_HighIQ[1]))
        p2 = scipy.stats.binom_test(Nmut_HighIQ[2], Nmut_HighIQ[2]+Nmut_HighIQ[1], p=CDS_HighIQ[2]/(CDS_HighIQ[2]+CDS_HighIQ[1]))
    print("NighIQ proband binom P:", p1, p2)
    if Null == "All":
        p1 = scipy.stats.binom_test(Nmut_LowIQ[0], sum(Nmut_LowIQ), p=CDS_LowIQ[0]/sum(CDS_LowIQ))
        p2 = scipy.stats.binom_test(Nmut_LowIQ[2], sum(Nmut_LowIQ), p=CDS_LowIQ[2]/sum(CDS_LowIQ))
    elif Null == "NS":
        p1 = scipy.stats.binom_test(Nmut_LowIQ[0], Nmut_LowIQ[0]+Nmut_LowIQ[1], p=CDS_LowIQ[0]/(CDS_LowIQ[0]+CDS_LowIQ[1]))
        p2 = scipy.stats.binom_test(Nmut_LowIQ[2], Nmut_LowIQ[2]+Nmut_LowIQ[1], p=CDS_LowIQ[2]/(CDS_LowIQ[2]+CDS_LowIQ[1]))
    print("LowIQ proband binom P:", p1, p2)
    for Rate, group in zip(Rate_HighIQ, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        if NormRate:
            xx = Rate/TotalMutRate_HighIQ
        else:
            xx = Rate/NHighIQProband
        normedRate1.append(xx)
    for Rate, group in zip(Rate_LowIQ, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        if NormRate:
            xx = Rate/TotalMutRate_LowIQ
        else:
            xx = Rate/NLowIQProband
        normedRate2.append(xx)
    ax2.plot(normedRate1, color="red", label="higher IQ")
    ax2.plot(normedRate2, color="blue", label="lower IQ")
    #ax2.axhline(y=1, color="black", linestyle=":")
    ax2.legend()
    plt.show()

def GeneBinsBurden2(parts, Nbins=3, title="", NormRate=False, NHighIQProband = 1870, NLowIQProband = 538, Null="All"):
    ALLRate, HighIQRate, LowIQRate = [],[],[]
    TotalALLRate, TotalHighIQRate, TotalLowIQRate = 0,0,0
    TotalMuts_ALL, TotalMuts_HighIQ, TotalMuts_LowIQ = 0,0,0
    TotalCDS_ALL,TotalCDS_HighIQ,TotalCDS_LowIQ = 0,0,0
    for i in range(Nbins):
        withLGD_ALL = sum(x.Nmut for x in parts[i])
        withLGD_HighIQ = sum(x.Nmut for x in parts[i] if x.IQgt70)
        withLGD_LowIQ = sum(x.Nmut for x in parts[i] if x.IQlt70)
        CDS_ALL = sum([x.CDSLength for x in parts[i]])
        CDS_HighIQ = sum([x.CDSLength for x in parts[i] if x.IQgt70])
        CDS_LowIQ = sum([x.CDSLength for x in parts[i] if x.IQlt70])
        #CDS_HighIQ = sum([x.CDSLength for x in parts[i]])
        #CDS_LowIQ = sum([x.CDSLength for x in parts[i]])
        TotalMuts_ALL += withLGD_ALL; TotalMuts_HighIQ += withLGD_HighIQ; TotalMuts_LowIQ += withLGD_LowIQ 
        ALLRate.append(withLGD_ALL/CDS_ALL)
        HighIQRate.append(withLGD_HighIQ/CDS_HighIQ)
        LowIQRate.append(withLGD_LowIQ/CDS_LowIQ)
        #print(withLGD/sum([x.CDSLength for x in parts[i]]))
        #print(len(parts[i]), withLGD0, sum([x.CDSLength for x in parts[i]]))
        TotalCDS_ALL += CDS_ALL
        TotalCDS_HighIQ += CDS_HighIQ
        TotalCDS_LowIQ += CDS_LowIQ
    TotalMutRate_ALL = TotalMuts_ALL/TotalCDS_ALL
    TotalMutRate_HighIQ = TotalMuts_HighIQ/TotalCDS_HighIQ
    TotalMutRate_LowIQ = TotalMuts_LowIQ/TotalCDS_LowIQ
    Nmut_G1 = sum([x.Nmut for x in parts[0]])
    Nmut_G2 = sum([x.Nmut for x in parts[1]])
    Nmut_G3 = sum([x.Nmut for x in parts[2]])
    CDS_G1 = sum([x.CDSLength for x in parts[0]])
    CDS_G2 = sum([x.CDSLength for x in parts[1]])
    CDS_G3 = sum([x.CDSLength for x in parts[2]])
    #print(Nmut_G1, Nmut_G2, CDS_G1, CDS_G2, CDS_G1/(CDS_G1+CDS_G2))
    Nmut_ALL = Nmut_G1+Nmut_G2+Nmut_G3
    p1 = scipy.stats.binom_test(Nmut_G1, Nmut_ALL, p=CDS_G1/(CDS_G1+CDS_G2+CDS_G3))
    p2 = scipy.stats.binom_test(Nmut_G3, Nmut_ALL, p=CDS_G3/(CDS_G1+CDS_G2+CDS_G3))
    print("binom P:", p1, p2)
    normedRate0 = []
    for Rate, group in zip(ALLRate, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        xx = Rate/TotalMutRate_ALL
        normedRate0.append(xx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2), dpi=80)
    fig.suptitle(title)
    ax1.plot(normedRate0, color="black", label="all proband")
    ax1.axhline(y=1, color="black", linestyle=":")
    ax1.legend()
    normedRate1 = []
    normedRate2 = []
    for Rate, group in zip(HighIQRate, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        xx = Rate/TotalMutRate_HighIQ
        normedRate1.append(xx)
    for Rate, group in zip(LowIQRate, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        xx = Rate/TotalMutRate_LowIQ
        normedRate2.append(xx)
    ax2.plot(normedRate1, color="red", label="higher IQ")
    ax2.plot(normedRate2, color="blue", label="lower IQ")
    ax2.axhline(y=1, color="black", linestyle=":")
    ax2.legend()
    plt.show()

def ExonBinsBurden(parts, Nbins=3, Null="All", NormRate=False, NHighIQProband = 1870, NLowIQProband = 638, title=""):
    Nmut_ALL, Nmut_HighIQ, Nmut_LowIQ = [],[],[]
    CDS_ALL, CDS_HighIQ, CDS_LowIQ = [],[],[]
    Rate_ALL, Rate_HighIQ, Rate_LowIQ = [],[],[]
    #TotalALLRate, TotalHighIQRate, TotalLowIQRate = 0,0,0
    #TotalMuts_ALL, TotalMuts_HighIQ, TotalMuts_LowIQ = 0,0,0
    #TotalCDS_ALL,TotalCDS_HighIQ,TotalCDS_LowIQ = 0,0,0
    for i in range(Nbins):
        Nmut_ALL.append(sum([x.Nmut for x in parts[i]]))
        Nmut_HighIQ.append(sum([x.Nmut for x in parts[i] if x.IQgt70]))
        Nmut_LowIQ.append(sum([x.Nmut for x in parts[i] if x.IQlt70]))
        CDS_ALL.append(sum([x.CDSLength for x in parts[i]]))
        #CDS_HighIQ.append(sum([x.CDSLength for x in parts[i] if x.IQgt70]))
        #CDS_LowIQ.append(sum([x.CDSLength for x in parts[i] if x.IQlt70]))
        CDS_HighIQ.append(sum([x.CDSLength for x in parts[i]]))
        CDS_LowIQ.append(sum([x.CDSLength for x in parts[i]]))
        Rate_ALL.append(Nmut_ALL[i]/CDS_ALL[i])
        Rate_HighIQ.append(Nmut_HighIQ[i]/CDS_HighIQ[i])
        Rate_LowIQ.append(Nmut_LowIQ[i]/CDS_LowIQ[i])
    TotalMutRate_ALL = sum(Nmut_ALL)/sum(CDS_ALL)
    TotalMutRate_HighIQ = sum(Nmut_HighIQ)/sum(CDS_HighIQ)
    TotalMutRate_LowIQ = sum(Nmut_LowIQ)/sum(CDS_LowIQ)
    if Null == "All":
        p1 = scipy.stats.binom_test(Nmut_ALL[0], sum(Nmut_ALL), p=CDS_ALL[0]/sum(CDS_ALL)) # All as null
        p2 = scipy.stats.binom_test(Nmut_ALL[2], sum(Nmut_ALL), p=CDS_ALL[2]/sum(CDS_ALL)) # All as null
    elif Null == "NS":
        #p1 = scipy.stats.binom_test(Nmut_ALL[0], Nmut_ALL[0]+Nmut_ALL[1], p=CDS_ALL[0]/(CDS_ALL[0]+CDS_ALL[1])) # NS as null
        #p2 = scipy.stats.binom_test(Nmut_ALL[2], Nmut_ALL[2]+Nmut_ALL[1], p=CDS_ALL[2]/(CDS_ALL[2]+CDS_ALL[1])) # NS as null
        p1 = scipy.stats.binom_test(Nmut_ALL[0], CDS_ALL[0], p=Nmut_ALL[1]/CDS_ALL[1]) # NS as null
        p2 = scipy.stats.binom_test(Nmut_ALL[2], CDS_ALL[2], p=Nmut_ALL[1]/CDS_ALL[1]) # NS as null
    print("All proband binom P:", p1, p2)
    normedRate0 = []
    for Rate, group in zip(Rate_ALL, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        if NormRate:
            xx = Rate/TotalMutRate_ALL
        else:
            xx = Rate/(NHighIQProband+NLowIQProband)
        normedRate0.append(xx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2), dpi=80)
    fig.suptitle(title)
    ax1.plot(normedRate0, color="black", label="all proband")
    #ax1.axhline(y=1, color="black", linestyle=":")
    ax1.legend()
    normedRate1 = []
    normedRate2 = []
    if Null == "All":
        p1 = scipy.stats.binom_test(Nmut_HighIQ[0], sum(Nmut_HighIQ), p=CDS_HighIQ[0]/sum(CDS_HighIQ))
        p2 = scipy.stats.binom_test(Nmut_HighIQ[2], sum(Nmut_HighIQ), p=CDS_HighIQ[2]/sum(CDS_HighIQ))
    elif Null == "NS":
        p1 = scipy.stats.binom_test(Nmut_HighIQ[0], Nmut_HighIQ[0]+Nmut_HighIQ[1], p=CDS_HighIQ[0]/(CDS_HighIQ[0]+CDS_HighIQ[1]))
        p2 = scipy.stats.binom_test(Nmut_HighIQ[2], Nmut_HighIQ[2]+Nmut_HighIQ[1], p=CDS_HighIQ[2]/(CDS_HighIQ[2]+CDS_HighIQ[1]))
    print("NighIQ proband binom P:", p1, p2)
    if Null == "All":
        p1 = scipy.stats.binom_test(Nmut_LowIQ[0], sum(Nmut_LowIQ), p=CDS_LowIQ[0]/sum(CDS_LowIQ))
        p2 = scipy.stats.binom_test(Nmut_LowIQ[2], sum(Nmut_LowIQ), p=CDS_LowIQ[2]/sum(CDS_LowIQ))
    elif Null == "NS":
        p1 = scipy.stats.binom_test(Nmut_LowIQ[0], Nmut_LowIQ[0]+Nmut_LowIQ[1], p=CDS_LowIQ[0]/(CDS_LowIQ[0]+CDS_LowIQ[1]))
        p2 = scipy.stats.binom_test(Nmut_LowIQ[2], Nmut_LowIQ[2]+Nmut_LowIQ[1], p=CDS_LowIQ[2]/(CDS_LowIQ[2]+CDS_LowIQ[1]))
    print("LowIQ proband binom P:", p1, p2)
    for Rate, group in zip(Rate_HighIQ, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        if NormRate:
            xx = Rate/TotalMutRate_HighIQ
        else:
            xx = Rate/NHighIQProband
        normedRate1.append(xx)
    for Rate, group in zip(Rate_LowIQ, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        if NormRate:
            xx = Rate/TotalMutRate_LowIQ
        else:
            xx = Rate/NLowIQProband
        normedRate2.append(xx)
    ax2.plot(normedRate1, color="red", label="higher IQ")
    ax2.plot(normedRate2, color="blue", label="lower IQ")
    #ax2.axhline(y=1, color="black", linestyle=":")
    ax2.legend()
    plt.show()

def ExonBinsBurden2(parts, Nbins=3, title=""):
    Rates0 = []
    Rates1 = []
    Rates2 = []
    OveralRate0,OveralRate1,OveralRate2 = 0,0,0
    TotalMuts0, TotalMuts1, TotalMuts2 = 0,0,0
    TotalCDS = 0
    for i in range(Nbins):
        withLGD0 = sum([x.Nmut for x in parts[i]])
        withLGD1 = sum([x.Nmut for x in parts[i] if x.IQgt70])
        withLGD2 = sum([x.Nmut for x in parts[i] if x.IQlt70])

        TotalMuts0 += withLGD0
        TotalMuts1 += withLGD1
        TotalMuts2 += withLGD2

        Rates0.append(withLGD0/sum([x.CDSLength for x in parts[i]]))
        Rates1.append(withLGD1/sum([x.CDSLength for x in parts[i]]))
        Rates2.append(withLGD2/sum([x.CDSLength for x in parts[i]]))

        TotalCDS += sum([x.CDSLength for x in parts[i]])
        #print(withLGD/sum([x.CDSLength for x in parts[i]]))
        print(len(parts[i]), withLGD0, sum([x.CDSLength for x in parts[i]]))
    OveralRate0 = TotalMuts0/TotalCDS
    OveralRate1 = TotalMuts1/TotalCDS
    OveralRate2 = TotalMuts2/TotalCDS
    Nmut_G1 = sum([x.Nmut for x in parts[0]])
    Nmut_G2 = sum([x.Nmut for x in parts[1]])
    Nmut_G3 = sum([x.Nmut for x in parts[2]])
    CDS_G1 = sum([x.CDSLength for x in parts[0]])
    CDS_G2 = sum([x.CDSLength for x in parts[1]])
    CDS_G3 = sum([x.CDSLength for x in parts[2]])
    print(Nmut_G1, Nmut_G2, CDS_G1, CDS_G2, CDS_G1/(CDS_G1+CDS_G2))
    p1 = scipy.stats.binom_test(Nmut_G1, Nmut_G1 + Nmut_G2, p=CDS_G1/(CDS_G1+CDS_G2))
    p2 = scipy.stats.binom_test(Nmut_G3, Nmut_G3 + Nmut_G2, p=CDS_G3/(CDS_G3+CDS_G2))
    print("binom P:", p1, p2)
    normedRate0 = []
    for Rate, group in zip(Rates0, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        xx = Rate/OveralRate0
        normedRate0.append(xx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2), dpi=80)
    fig.suptitle(title)
    ax1.plot(normedRate0, color="black", label="all probands")
    ax1.axhline(y=1, color="black", linestyle=":")
    ax1.legend()
    normedRate1 = []
    normedRate2 = []
    for Rate, group in zip(Rates1, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        xx = Rate/OveralRate1
        normedRate1.append(xx)
    for Rate, group in zip(Rates2, ["Prenatal bias","Unsignificant bias","Postnatal bias"]):
        xx = Rate/OveralRate2
        normedRate2.append(xx)
    ax2.plot(normedRate1, color="red", label="higher IQ")
    ax2.plot(normedRate2, color="blue", label="lower IQ")
    ax2.axhline(y=1, color="black", linestyle=":")
    ax2.legend()
    ax2.legend()
    plt.show()

##############################################################################
# Burden of Developmental stages
##############################################################################
SVIPGenes = ["ASXL3","CHD2","CHD8","DSCAM","DYRK1A","FOXP1","HIVEP2","SCN2A","ADNP","CHAMP1","CSNK2A1","GRIN2B",
        "HNRNPH2","MED13L","PACS1","PPP2R5D","SETBP1","STXBP1","SYNGAP1"]
def LoadGenCodeTrans():
    Gene2Trans = {}
    hand = open("unifiedmodel/VIPgenes.gencode.v19.gtf", 'rt')
    for l in hand:
        if l.startswith("#"):
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        CHR = llist[0].lstrip("chr")
        strand = llist[6]
        start = (llist[3])
        end = (llist[4])
        if llist[2] == "transcript" and info["transcript_type"] == "protein_coding":
            Gene2Trans[info["gene_name"]] = info["transcript_id"]
    Gene2Trans["ASXL3"]="ENST00000269197.5"
    Gene2Trans["CHD8"]="ENST00000399982.2"
    Gene2Trans["DYRK1A"]="ENST00000339659.4"
    Gene2Trans["CHD2"]="ENST00000394196.4"
    Gene2Trans["SCN2A"]="ENST00000357398.3"
    Gene2Trans["FOXP1"]="ENST00000318789.4"
    Gene2Trans["HIVEP2"]="ENST00000367603.2"
    Gene2Trans["PPP2R5D"]="ENST00000485511.1"
    Gene2Trans["HIVEP2"]="ENST00000367603.2"
    Gene2Trans["SYNGAP1"]="ENST00000418600.2"
    Gene2Trans["STXBP1"]="ENST00000373302.3"
    Gene2Trans["PACS1"]="ENST00000320580.4"
    Gene2Trans["MED13L"]="ENST00000281928.3"
    Gene2Trans["CHAMP1"]="ENST00000361283.1"
    Gene2Trans["SETBP1"]="ENST00000282030.5"
    Gene2Trans["CSNK2A1"]="ENST00000217244.3"
    Gene2Trans["ADNP"]="ENST00000396029.3"
    Gene2Trans["DSCAM"]="ENST00000400454.1"
    Gene2Trans["HNRNPH2"]="ENST00000400454.1"
    return Gene2Trans

def LoadSSCCohort():
    Jiayao_features = pd.read_csv("unifiedmodel/features.jiayao.65.csv")
    df = Jiayao_features
    for i, row in df.iterrows():
        df.loc[i, "sfari_id"] = df.loc[i, "familyId"]
        df.loc[i, "sex"] = df.loc[i, "gender"]
        Chr, Pos, Ref, Alt = row["vcfVariant"].split(":")
        Len = max(0, len(Ref) - len(Alt))
        df.loc[i, "location"] = "{}:{}-{}".format(Chr, Pos, int(Pos)+Len)
        df.loc[i, "inheritance_status"] = "de-novo"
        df.loc[i, "genetic_status"] = row["effectGene"]
        df.loc[i, "composite_standard_score"] = row["VABS"]
        df.loc[i, "Cohort"] = "SSC"
    SSC = df[["Cohort", "sfari_id","sex","genetic_status","inheritance_status","composite_standard_score","location"]]
    return SSC

def LoadVIPSingleGeneData(DIR):
    dfs1 = []
    dfs2 = []
    for gene in SVIPGenes:
        try:
            df1 = pd.read_csv("/".join([DIR,gene,"vineland_ii.csv"]))
            dfs1.append(df1)
        except:
            pass
        try:
            df2 = pd.read_csv("/".join([DIR,gene,"lab_results.csv"]))
            dfs2.append(df2)
        except:
            pass
    VABS_DF = pd.concat(dfs1)
    LGD_DF = pd.concat(dfs2)
    LGD_DF = LGD_DF[["sfari_id", "relationship_to_iip", "genetic_status","lab_results_single_hgvs"]]
    LGD_DF = LGD_DF[LGD_DF["relationship_to_iip"]=="Initially identified proband"]
    VABS_DF = VABS_DF[VABS_DF["relationship_to_iip"]=="Initially identified proband"]
    LGD_DF = LGD_DF.dropna()
    VABS_DF = VABS_DF[["sfari_id","sex","genetic_status","inheritance_status", "composite_standard_score"]]
    VABS_DF = VABS_DF.reset_index(drop=True)
    return VABS_DF, LGD_DF
    
def addHGVS(row, Gene2Trans):
    Trans = Gene2Trans[row["genetic_status"]]
    hgvs = row["lab_results_single_hgvs"]
    hgvs = "c.{}".format(hgvs) if not (hgvs.startswith("c.")) else hgvs
    return "{}:{}".format(Trans, hgvs)

def CleanUpSVIPVEPData():
    VEP = pd.read_csv("./data/SVIP.V6.Genes.HGVS.VEP.tsv", delimiter="\t")
    Transcripts = [x for x in Gene2Trans.values()]
    #LGDs = ['frameshift_variant', 'stop_gained', 'splice_donor_variant']
    LGDs = VEP_LGD
    VEP = VEP[(VEP["Feature"].isin(Transcripts)) & (VEP["Consequence"].isin(LGDs))]
    VEP = VEP.drop_duplicates(subset="#Uploaded_variation", keep='first')
    VEP.set_index("#Uploaded_variation", inplace=True)
    VEP.to_csv("data/SVIP.V6.VEP.RecGenes.LGD.txt", sep="\t")

def serachVEP(row, vip_hgvs, Gene2Trans, VEP):
    transcript = Gene2Trans[row["genetic_status"]]
    vip_hgvs = "c.{}".format(vip_hgvs) if not (vip_hgvs.startswith("c.")) else vip_hgvs
    key = "{}:{}".format(transcript, vip_hgvs)
    location = VEP.loc[key, "Location"]
    return location

def AnnotateVEP2VABSDF(VABS_DF, LGD_DF, Gene2Trans, VEP):
    for i, row in VABS_DF.iterrows():
        spid, gene = row["sfari_id"], row["genetic_status"]
        vip_hgvs = LGD_DF[(LGD_DF["sfari_id"]==spid) & ((LGD_DF["genetic_status"]==gene))]
        if vip_hgvs.shape[0] == 1:
            vip_hgvs = vip_hgvs["lab_results_single_hgvs"].values[0]
            try:
                VABS_DF.at[i, "location"] = serachVEP(row, vip_hgvs, Gene2Trans, VEP)
            except KeyError:
                #print(vip_hgvs)
                VABS_DF.at[i, "location"] = np.nan
        else:
            VABS_DF.at[i, "location"] = np.nan
        VABS_DF.at[i, "sex"] = row["sex"][0].upper()
    VABS_DF = VABS_DF.dropna()
    return VABS_DF



##############################################################################
# Phenotypic Independence
##############################################################################
def RatioPermutationTest(Pairs, Phenotype1="NVIQ", Phenotype2="VABS", N1 = 60, N2 = 8, Npermute=1000):
    Ratios, SelectedSamples = [],[]
    i,j = 0,0
    while i < Npermute:
        j += 1
        sp = np.random.choice(Pairs, N1)
        First8 = sp[:N2]
        Last60 = sp[N2:]
        First8_NVIQ_DIFF = np.mean([x.NVIQ_DIFF for x in First8])
        First8_VABS_DIFF = np.mean([x.VABS_DIFF for x in First8])
        Last60_NVIQ_DIFF = np.mean([x.NVIQ_DIFF for x in Last60])
        Last60_VABS_DIFF = np.mean([x.VABS_DIFF for x in Last60])
        NVIQ_Ratio = First8_NVIQ_DIFF/Last60_NVIQ_DIFF
        #print(NVIQ_Ratio)
        if NVIQ_Ratio <= 0.4:
            i += 1
            Ratios.append(First8_VABS_DIFF/Last60_VABS_DIFF)
            SelectedSamples.append(sp)
    return Ratios, SelectedSamples

def PlotHistAndP(Null, Obs):
    n, bins, patches = plt.hist(Null, bins=20)
    p = get_smaller_P(Obs, [0] + Null)
    plt.vlines(x=Obs, ymin=0, ymax=max(n))
    plt.text(x=Obs, y=max(n), s="p=%.3f"%p)
    plt.show()


def RatioPermutationTestNVIQvsVIQ(Pairs, N1 = 60, N2 = 8, Npermute=1000):
    Ratios, SelectedSamples = [],[]
    i,j = 0,0
    while i < Npermute:
        j += 1
        sp = np.random.choice(Pairs, 60)
        First8 = sp[:8]
        Last60 = sp[8:]
        First8_NVIQ_DIFF = np.mean([x.NVIQ_DIFF for x in First8])
        First8_VIQ_DIFF = np.mean([x.VIQ_DIFF for x in First8])
        Last60_NVIQ_DIFF = np.mean([x.NVIQ_DIFF for x in Last60])
        Last60_VIQ_DIFF = np.mean([x.VIQ_DIFF for x in Last60])
        NVIQ_Ratio = First8_NVIQ_DIFF/Last60_NVIQ_DIFF
        #print(NVIQ_Ratio)
        if NVIQ_Ratio <= 0.4:
            i += 1
            Ratios.append(First8_VIQ_DIFF/Last60_VIQ_DIFF)
            SelectedSamples.append(sp)
    return Ratios, SelectedSamples

def RatioPermutationTestNVIQvsVIQ(Pairs, Phenotype1="NVIQ", Phenotype2="VABS", N1 = 60, N2 = 8, Npermute=1000):
    Ratios, SelectedSamples = [],[]
    i,j = 0,0
    while i < Npermute:
        j += 1
        sp = np.random.choice(Pairs, 60)
        First8 = sp[:N2]
        Last60 = sp[N2:]
        First8_NVIQ_DIFF = np.mean([x.NVIQ_DIFF for x in First8])
        First8_VABS_DIFF = np.mean([x.VABS_DIFF for x in First8])
        Last60_NVIQ_DIFF = np.mean([x.NVIQ_DIFF for x in Last60])
        Last60_VABS_DIFF = np.mean([x.VABS_DIFF for x in Last60])
        NVIQ_Ratio = First8_NVIQ_DIFF/Last60_NVIQ_DIFF
        #print(NVIQ_Ratio)
        if NVIQ_Ratio <= 0.4:
            i += 1
            Ratios.append(First8_VABS_DIFF/Last60_VABS_DIFF)
            SelectedSamples.append(sp)
    return Ratios, SelectedSamples


##############################################################################
# Frac. Isoform - normalized pheno
##############################################################################
def LoadGTF4FracIso(GTFFile):
    Genes = {}
    hand = open(GTFFile, 'rt')
    for l in hand:
        if l.startswith("#"):
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        CHR = llist[0].lstrip("chr")
        strand = llist[6]
        start = int(llist[3])
        end = int(llist[4])
        if llist[2] == "gene":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            if gene_name not in Genes:
                Genes[gene_name] = GTFGene(gene_name, gene_id, strand)
        elif llist[2] == "transcript":
            #print(info["transcript_id"], info["transcript_type"])
            gene_name = info["gene_name"]
            gene_id = info["gene_id"]
            transcript_name = info["transcript_id"]
            transcript_id = info["transcript_id"].split(".")[0]
            transcript_type = info["transcript_type"]
            if transcript_id not in Genes[gene_name].Transcripts and transcript_type=="protein_coding":
                #print("XX", info["transcript_id"], info["transcript_type"])
                #print(gene_name, Genes[gene_name].Transcripts.keys())
                Genes[gene_name].Transcripts[transcript_id] = GTFTranscript(gene_name, transcript_name, transcript_id, strand)
                #print(gene_name, Genes[gene_name].Transcripts.keys())
        elif llist[2] == "exon":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"]
            exon_id = info["exon_id"]
            transcript_name = info["transcript_id"]
            transcript_id = info["transcript_id"].split(".")[0]
            transcript_type = info["transcript_type"]
            if transcript_type=="protein_coding":
                exon= GTFExon(exon_id, start, end, transcript_id, strand)
                #Genes[gene_name].Transcripts[transcript_id].Exons.append(exon)
                Genes[gene_name].Transcripts[transcript_id].Exons[exon_id] = exon
    return Genes

def searchExon4FracIso(Gene, Pos, Ref, Alt, Genes):
    Pos, LenV = int(Pos), len(Ref)-len(Alt)
    gene_obj = Genes[Gene]
    _Exons, Transcripts = [],[]
    for transid, transobj in gene_obj.Transcripts.items():
        for exon in transobj.Exons.values():
            if Pos > exon.start -3 and Pos < exon.end +3:
                _Exons.append(exon.ExonID)
                Transcripts.append(transid)
                break
            elif LenV > 0:
                if (Pos < exon.start-3 and Pos + LenV > exon.start ) or (Pos < exon.end and Pos + LenV > exon.end +3):
                    _Exons.append(exon.ExonID)
                    Transcripts.append(transid)
                    break
    return list(set(_Exons)), list(set(Transcripts))

def LoadVar4FracIso(DF, Genes):
    Jiayao_features = DF
    #Jiayao_features = pd.read_csv("/Users/jiayao/Work/BrainDisorders/src/data/SVIP.V4.RecGenes.LGD.txt", delimiter="\t")
    #Jiayao_features.loc[Jiayao_features["effectGene"]=="MLL5", "effectGene"] = "KMT2E"
    Jiayao_features["Exons"] = ""
    Jiayao_features["Transcripts"] = ""
    for i, row in Jiayao_features.iterrows():
        #famid, gene, (Chr, Pos, Ref, Alt) = row["familyId"], row["effectGene"], row["vcfVariant"].split(":")
        famid, gene, (Chr, Pos, Ref, Alt) = row["familyId"], row["effectGene"], row["vcfVariant"].split(":")
        ExonIDs, TranscriptIDs = searchExon4FracIso(gene, Pos, Ref, Alt, Genes)
        Jiayao_features.at[i, "Exons"] = ExonIDs
        Jiayao_features.at[i, "Transcripts"] = TranscriptIDs
    return Jiayao_features

def LoadVar4FracIso2(DF, Genes):
    Jiayao_features = DF #pd.read_csv("/Users/jiayao/Work/BrainDisorders/src/data/SVIP.V4.RecGenes.LGD.txt", delimiter="\t")
    Jiayao_features.loc[Jiayao_features["genetic_status"]=="MLL5", "genetic_status"] = "KMT2E"
    Jiayao_features["Exons"] = ""
    Jiayao_features["Transcripts"] = ""
    for i, row in Jiayao_features.iterrows():
        #famid, gene, (Chr, Pos, Ref, Alt) = row["familyId"], row["effectGene"], row["vcfVariant"].split(":")
        Jiayao_features.loc[i, "effectGene"] = row["genetic_status"]
        gene = row["genetic_status"]
        Chr, Pos = row["location"].split(":")
        Pos, Pos2 = Pos.split("-")
        Ref = "A"
        Alt = "A" * (len(Pos2) - len(Pos) + 1)
        ExonIDs, TranscriptIDs = searchExon4FracIso(gene, Pos, Ref, Alt, Genes)
        Jiayao_features.at[i, "Exons"] = ";".join(ExonIDs)
        Jiayao_features.at[i, "Transcripts"] = ";".join(TranscriptIDs)
    return Jiayao_features

def SelectTissue(GTEX_EXP_MATRIX):
    pass 

def SummarizeTMPCrossSample(GTEX_EXP_MATRIX, transform="linear", method="mean"):
    Res = {}
    for i, row in GTEX_EXP_MATRIX.iterrows():
        EXPs = list(row[2:])
        #EnsGene, EnsTrans = list(row[0:2])
        EnsTrans, EnsGene = list(row[0:2])
        EnsGene = EnsGene.split(".")[0]
        EnsTrans = EnsTrans.split(".")[0]
        if EnsGene not in Res:
            Res[EnsGene] = {}
        if transform == 'log2':
            EXPs = [math.log2(exp+1) for exp in EXPs]
        if method == "mean":
            Value = np.mean(EXPs)
        elif method == "median":
            Value = np.median(EXPs)
        Res[EnsGene][EnsTrans] = Value
    return Res

def SharedFracIsoNormPheno(DF, GeneSYM2ID, GeneTransTPM, pheno="NVIQ", Norm=False, SameExon=False):
    pheno_diffs = []
    FracIsoComs = []
    for gene in list(set(DF["effectGene"].values)):
        df = DF[(DF["effectGene"]==gene)]
        EnsGeneID = GeneSYM2ID[gene]
        pheno_diff_gene = []
        for row1, row2 in itertools.combinations(df.iterrows(), r=2):
            row1 = row1[1]
            row2 = row2[1]
            if not SameExon:
                if row1["Exons"] == row2["Exons"]:
                    continue
            #if pheno == "composite_standard_score":
            #    Y1 = row1[pheno]
            Y1 = max(0, 100-row1[pheno])
            Y2 = max(0, 100-row2[pheno])
            Iso1 = row1["Transcripts"].split(";")
            Iso2 = row2["Transcripts"].split(";")
            Frac1 = len(set(Iso1))/len( GeneTransTPM[EnsGeneID].keys() )
            Frac2 = len(set(Iso2))/len( GeneTransTPM[EnsGeneID].keys() )
            if Norm:
                try:
                    Yn1, Yn2 = Y1/Frac1, Y2/Frac2
                except ZeroDivisionError:
                    #print(Frac1, Frac2, row1, row2)
                    continue
            else:
                Yn1, Yn2 = Y1, Y2
            pheno_diff = abs(Yn1-Yn2)
            FracIsoCom = len(set(Iso1).intersection(set(Iso2))) / len( GeneTransTPM[EnsGeneID].keys() )
            FracIsoComs.append(FracIsoCom)
            pheno_diff_gene.append(pheno_diff)
        pheno_diffs.extend(pheno_diff_gene)
    return FracIsoComs, pheno_diffs


def Permute(DF, GeneSYM2ID, GeneTransTPM, N=1000, pheno="NVIQ"):
    pearson_rhos = []
    spearman_rhos = []
    tmp_df = DF.copy(deep=True)
    for i in range(N):
        tmp_df[pheno] = np.random.permutation(tmp_df[pheno].values)
        #display(tmp_df.head(2))
        FracIsoComs, pheno_diffs = SharedFracIsoNormPheno(tmp_df, GeneSYM2ID, GeneTransTPM, pheno=pheno)
        r1, p1 = pearsonr(FracIsoComs, pheno_diffs)
        r2, p2 = spearmanr(FracIsoComs, pheno_diffs)
        pearson_rhos.append(r1)
        spearman_rhos.append(r2)
    return pearson_rhos, spearman_rhos

def PlotSharedIso(FracIsoComs, pheno_diffs):
    plt.scatter(FracIsoComs, pheno_diffs)
    plt.xlabel("Frac.Trans.Affected")
    plt.ylabel("Normed.NVIQ.Diff")

    (r1, p1) = (pearsonr(FracIsoComs, pheno_diffs))
    (r2, p2) = (spearmanr(FracIsoComs, pheno_diffs))
    plt.text(x=max(FracIsoComs)*0.5, y=max(pheno_diffs)*0.9, s="pearsonr=%.2f p=%.2e\nspearmanr=%.2f p=%.2e"%(r1,p1,r2,p2))
    plt.show()

def GroupTest(FracIsoComs, IQ_diffs, cutoff=0.5):
    Group1, Group2 = [],[]
    for frac, diff in zip(FracIsoComs, IQ_diffs):
        if frac > cutoff:
            Group2.append(diff)
        else:
            Group1.append(diff)
    print(len(Group1), len(Group2))
    print(np.mean(Group1), np.mean(Group2))
    print(scipy.stats.mannwhitneyu(Group1, Group2))

def LoadGenesForSVIPSSC():
    Genes = {}
    hand = open("unifiedmodel/RecLGDgenes.gencode.v19.gtf", 'rt')
    GeneIDs = []
    for l in hand:
        if l.startswith("#"):
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        CHR = llist[0].lstrip("chr")
        strand = llist[6]
        start = int(llist[3])
        end = int(llist[4])
        if llist[2] == "gene":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            Genes[gene_name] = GTFGene(gene_name, gene_id, strand)
            GeneIDs.append(gene_id)

    hand = open("unifiedmodel/VIPgenes.gencode.v19.gtf", 'rt')
    for l in hand:
        if l.startswith("#"):
            continue
        llist = l.strip().split("\t")
        info = gtf_info_parser(llist[8])
        CHR = llist[0].lstrip("chr")
        strand = llist[6]
        start = int(llist[3])
        end = int(llist[4])
        if llist[2] == "gene":
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            Genes[gene_name] = GTFGene(gene_name, gene_id, strand)
            GeneIDs.append(gene_id)

    GeneIDs = set(GeneIDs)
    return GeneIDs, Genes

##############################################################################
# NMD Efficiency modeling using GTEX data 
##############################################################################
def CollectASECount(_file):
    ASE_count_dir = "/Users/jiayao/Work/BrainDisorders/data/GTEx/Andy/INDV/"
    writer = csv.writer(open("data/GTEX_ASE_SNP.3.tsv", 'wt'), delimiter="\t")
    for filename in os.listdir(ASE_count_dir):
        if filename.endswith(".pp"):
            reader = csv.reader(open(ASE_count_dir+filename, 'rt'), delimiter="\t")
            header = next(reader)
            for row in reader:
                SAMPLE = row[1]
                ID = row[2]
                INFO = row[7]
                DEPTH = int(row[8])
                NREF = int(row[11])
                NALT = int(row[12])
                #if NREF + NALT >= 8:
                #if DEPTH >= 8:
                writer.writerow([SAMPLE, ID, INFO, NREF, NALT])

class ASEVar:
    def __init__(self, SAMPLE, ID, INFO, NREF, NALT, GENE):
        self.sample = SAMPLE
        self.id = ID
        self.vcfid = INFO
        self.nref = NREF
        self.nalt = NALT
        self.gene = GENE

class ASEGene:
    def __init__(self, gene):
        self.ID = gene
        self.ASEVars = []
        self.Nobs = 0
        self.Nvar = 0

def TissueGeneMut(Mutable, GTExMeta, Tissue):
    SAMPID = set(GTExMeta[GTExMeta["SMTS"]==Tissue]["SAMPID"].values)
    mutable = Mutable[Mutable["SAMPLE"].isin(SAMPID)]
    Genes = set(mutable["SEVERE_GENE"].values)
    res = {}
    for i, row in mutable.iterrows():
        SAMPLE, ID, INFO, NREF, NALT, GENE = row["SAMPLE"], row["ID"], row["INFO"], row["NREF"], row["NALT"], row["SEVERE_GENE"]
        var = ASEVar(SAMPLE, ID, INFO, NREF, NALT, GENE)
        if GENE not in res:
            res[GENE] = []
        res[GENE].append(var)
    return res

def TissueLGD(Mutable, GTExMeta, Tissue):
    SAMPID = set(GTExMeta[GTExMeta["SMTS"]==Tissue]["SAMPID"].values)
    mutable = Mutable[Mutable["SAMPLE"].isin(SAMPID)]
    res = {}
    for i, row in mutable.iterrows():
        SAMPLE, ID, INFO, NREF, NALT, GENE = row["SAMPLE"], row["ID"], row["INFO"], row["NREF"], row["NALT"], row["SEVERE_GENE"]
        var = ASEVar(SAMPLE, ID, INFO, NREF, NALT, GENE)
        res[ID] = var
    return res

def Jacobian1(a, b):
    return math.pow((a+b), -5/2)

def AB2Search(a, b):
    return math.log(a/b), math.log(a+b)


def Jacobian2(a, b):
    #return (math.exp(2*v+u)) / math.pow((math.exp(u)+1), 2)
    return a*b #*math.pow((a+b), -5/2) #math.log(a/b), math.log(a+b)

def SamplingAlphaBeta(step = 1):
    logMeans = np.arange(-10, 10+step, step)
    logVars = np.arange(-10, 10+step, step)
    #Qs = [scipy.stats.halfcauchy(x) for x in STDs]
    #return Means, STDs, QSTDs
    a_b_q = []
    for logmu in logMeans:
        for logvar in logVars:
            a,b = reparameter(logmu, logvar)
            q = math.log(scipy.stats.halfcauchy.pdf(1/math.sqrt(a+b)))
            a_b_q.append([a,b,q])
    return a_b_q

def reparameter(u_, v_):
    expu, expv = math.exp(u_), math.exp(v_)
    a = (expu * expv) / (1 + expu)
    b = expv - a
    return a, b

def beta_binom_ASE(f, a, b, k, n):
    res1 = beta.logpdf(f, a,b)
    res2 = binom.logpmf(k, n, f)
    return math.exp(res1 + res2)

def beta_binom_ASE2(f, a, b, k, n):
    res1 = beta.logpdf(f, a,b)
    res2 = binom.logpmf(k, n, f)
    return res1, res2 #math.exp(res1 + res2)

def ComputeLikelihood(genedat, alpha, beta, Q):
    posteriori = 0
    for k, var in enumerate(genedat):
        Kvi, Nvi = var.nalt, var.nref+var.nalt
        res, err = quad(beta_binom_ASE, 0, 1, args=(alpha, beta, Kvi, Nvi), )
        #res = SimpsonQuad(beta_binom_ASE, 0, 1, args=(alpha, beta, Kvi, Nvi), steps=2500)
        res = math.log(res)
        posteriori += res 
        #print(Kvi, Nvi, posteriori)
    posteriori += scipy.stats.halfcauchy.logpdf(1/math.sqrt(alpha+beta))
    posteriori += math.log(Jacobian1(alpha, beta))
    posteriori += math.log(Jacobian2(alpha, beta))
    return posteriori

def GridSearchASE(a_b_q, genedat, dx=0.001):
    max_posteriori, _alpha, _beta = -np.inf,0,0
    res = []
    for i, (alpha, beta, QAB) in enumerate(a_b_q):
        #print(alpha, beta)
        posteriori = ComputeLikelihood(genedat, alpha, beta, QAB)
        if posteriori > max_posteriori:
            max_posteriori = posteriori; _alpha = alpha; _beta = beta
        if i % 10 == 0:
            sys.stdout.write("\r %d computed"%i)
        res.append([(alpha, beta), posteriori])
        #print(math.log(Jacobian1(alpha, beta)), math.log(Jacobian2(alpha, beta)), QAB)
    return res#max_posteriori, _alpha, _beta

def P_F_E(f,e):
    return f*(1-e) / (1-f*e)

def beta_binom_NMD(e, f, ae, be, af, bf, k, n):
    res1 = beta.logpdf(e, ae, be)
    res2 = binom.logpmf(k, n, P_F_E(f,e))
    res3 = beta.logpdf(f, af, bf)
    return math.exp(res1 + res2 + res3)

def _halfcauchy(x):
    return -1 * math.log(1+x**2)

def ComputeLikelihoodNMDPara(LGDVarDat, alpha, beta, QAB):
    posteriori = 0
    #af, bf = LGDVarDat.af, LGDVarDat.bf
    for i, (k, N, af, bf) in enumerate(LGDVarDat):
        Kvi, Nvi = k, N
        #res, err = dblquad(beta_binom_NMD, 0, 1, lambda f:0, lambda f:1, args=(alpha, beta, af, bf, Kvi, Nvi), epsabs=1e-4, epsrel=1e-4)
        #res, err = dblquad(beta_binom_NMD, 0, 1, lambda f:0, lambda f:1, args=(alpha, beta, af, bf, Kvi, Nvi))
        res, err = dblquad(beta_binom_NMD, 0, 1, lambda f:0, lambda f:1, args=(alpha, beta, af, bf, Kvi, Nvi))
        #res = res * Jacobian1(alpha, beta)
        #u_, v_ = AB2Search(alpha, beta)
        #res = res * Jacobian2(u_, v_)
        posteriori += math.log(res)
        #sys.stderr.write("\r%d"%i)
        print(Kvi, Nvi, math.log(res))
    posteriori += _halfcauchy(1 / math.sqrt(alpha+beta)) 
    posteriori += math.log(Jacobian1(alpha, beta))
    posteriori += math.log(Jacobian2(alpha, beta))
    print(math.log(Jacobian1(alpha, beta)),  math.log(Jacobian2(alpha, beta)), _halfcauchy(1 / math.sqrt(alpha+beta)))
    return posteriori

def GridSearchNMD(a_b_q, LGDVarDat, dx=0.001):
    max_posteriori, _alpha, _beta = -np.inf,0,0
    for i, (alpha, beta, QAB) in enumerate(a_b_q):
        print(alpha, beta)
        posteriori = ComputeLikelihoodNMDPara(LGDVarDat, alpha, beta, QAB)
        print(math.log(Jacobian1(alpha, beta)),  math.log(Jacobian1(alpha, beta)), _halfcauchy(1 / math.sqrt(alpha+beta)))
        if posteriori > max_posteriori:
            max_posteriori = posteriori; _alpha = alpha; _beta = beta
        if i % 10 == 0:
            sys.stdout.write("\r %d computed"%i)
    return max_posteriori, _alpha, _beta

def LoadLGDDat(fname):
    hand = open(fname, 'rt')
    l = next(hand)
    res = []
    for l in hand:
        llist = l.strip().split(",")
        res.append([int(x) for x in llist] )
    return res


def beta_binom_Ehat(f, e, af, bf, k, n):
    res2 = binom.pmf(k, n, P_F_E(f,e))
    res3 = beta.pdf(f, af, bf)
    return res2 * res3

def ComputeLikelihoodEhat(var, e, af, bf, ae, be):
    posteriori = 0
    #f, af, bf, ae, be = LGDVarDat.f, LGDVarDat.af, LGDVarDat.bf, LGDVarDat.ae, LGDVarDat.be 
    Kvi, Nvi = var.nalt, var.nref+var.nalt
    res, err = quad(beta_binom_Ehat, 0, 1, args=(e, af, bf, Kvi, Nvi))
    #res = SimpsonQuad(beta_binom_Ehat, 0, 1, args=(e, af, bf, Kvi, Nvi))
    #print(math.log(res))
    posteriori += math.log(res)
    #print(e, ae, be, beta.pdf(e, ae, be))
    #posteriori += math.log(beta.pdf(e, ae, be))
    return posteriori

def EstimateE_hat(LGDVarDat, af, bf, ae, be, dx=1e-3):
    max_posteriori, e_hat = -np.inf, 0
    posterioris = []
    for i, e in enumerate(np.arange(dx,1,dx)):
        posteriori = ComputeLikelihoodEhat(LGDVarDat, e, af, bf, ae, be)
        posterioris.append(posteriori)
        if posteriori > max_posteriori:
            max_posteriori = posteriori; e_hat = e
        #if i % 1000 == 0:
        #    sys.stdout.write("\r %d computed"%i)
    return max_posteriori, e_hat, np.arange(dx,1,dx), posterioris 

def TestOneVar(Var, AEBE, Tissue="ADPS"):
    print(Var.sample)
    print(Var.gene)
    AEBE = AEBE[AEBE["tissue"]==Tissue]
    AE, BE = AEBE["a_hat"].values[0], AEBE["b_hat"].values[0]
    AFBF = pd.read_csv("/Users/jiayao/Work/BrainDisorders/src/NMD_model/TissueASE/{}.csv".format(Tissue))
    AFBF = AFBF[AFBF["gene"]==Var.gene]
    AF, BF = AFBF["ahat"].values[0], AFBF["bhat"].values[0]
    maxP, e_hat, all_e, allPosts = EstimateE_hat(Var, AF, BF, AE, BE, dx=1e-3)
    print(maxP, e_hat)
    plt.plot(all_e, allPosts)
    plt.show()

def SimpsonQuad(func, start, end, args, steps=100):
    step_size = (end - start ) / steps
    res = 0
    for step in range(steps):
        x = start + step_size/2
        _args = tuple([x] + [_ for _ in args])
        start += step_size
        h = func(*_args)
        res += step_size*h
    return res

def beta_binom_Ehat2(f, e, af, bf, k, n):
    fprime = P_F_E(f,e)
    res2 = binom.logpmf(k, n, fprime)
    #res2 = binom.pmf(k, n, f)
    res3 = beta.logpdf(f, af, bf)
    return fprime, res2, res3

def SimpsonQuad2(func, start, end, args, steps=10):
    step_size = (end - start ) / steps
    res = 0
    Dat = []
    for step in range(steps):
        x = start #+ step_size/2
        _args = tuple([x] + [_ for _ in args])
        start += step_size
        fp, binom, beta = func(*_args)
        try:
            #print(x, math.log(binom), math.log(beta),)
            #Dat.append([x, fp,  math.log(binom), math.log(beta)])
            Dat.append([x, fp,  binom, beta])
        except:
            #print(x, 0, 0)
            Dat.append([x, fp, 0, 0])
        h = binom * beta
        res += step_size*h
    return res, Dat

def ComputeLikelihoodEhat2(var, e, af, bf, ae, be):
    posteriori = 0
    Kvi, Nvi = var.nalt, var.nref+var.nalt
    print(Kvi, Nvi)
    res, Dat = SimpsonQuad2(beta_binom_Ehat2, 0, 1, args=(e, af, bf, Kvi, Nvi))
    posteriori += math.log(res)
    #posteriori += math.log(beta.pdf(e, ae, be))
    return posteriori, Dat

def EstimateE_hat2(LGDVarDat, af, bf, ae, be, dx=1e-3):
    max_posteriori, e_hat = -np.inf, 0
    posterioris = []
    for i, e in enumerate(np.arange(0,1,dx)):
        #print(e,),
        posteriori, Dat = ComputeLikelihoodEhat2(LGDVarDat, e, af, bf, ae, be)
        posterioris.append(posteriori)
        if posteriori > max_posteriori:
            max_posteriori = posteriori; e_hat = e
        #if i % 1000 == 0:
        #    sys.stdout.write("\r %d computed"%i)
        #print([e]+ Dat)
        for x, fp,  bi, be in Dat:
            print ("%.2f %.2f %.5f %.5f %.5f"%(e, x, fp, bi, be))
    return max_posteriori, e_hat, np.arange(dx,1,dx), posterioris 
