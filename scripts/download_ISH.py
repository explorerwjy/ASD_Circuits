import requests
import csv
import numpy as np
import argparse
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import pandas as pd


API_PATH = "http://api.brain-map.org/api/v2/data"
GRAPH_ID = 1
MOUSE_PRODUCT_ID = 1 # aba
PLANE_ID = 1 # coronal
TOP_N = 2000

DATA_SET_QUERY_URL = ("%s/SectionDataSet/query.json" +\
		 "?criteria=[failed$eq'false'][expression$eq'true']" +\
		 ",products[id$eq%d]" +\
		 ",plane_of_section[id$eq%d]") \
		 % (API_PATH, MOUSE_PRODUCT_ID, PLANE_ID)

UNIONIZE_FMT = "%s/StructureUnionize/query.json" +\
		"?criteria=[section_data_set_id$eq%d],structure[graph_id$eq1]" +\
		("&include=section_data_set(products[id$in%d])" % (MOUSE_PRODUCT_ID)) +\
		"&only=id,structure_id,sum_pixels,expression_energy,section_data_set_id" 

STRUCTURES_URL = ("%s/Structure/query.csv?" +\
		"criteria=[graph_id$eq%d]&numRows=all") \
		% (API_PATH, GRAPH_ID)
STRUCTURE_UNIONIZE_CONN_URL = "%s/ProjectionStructureUnionize/query.csv?criteria=[section_data_set_id$eq%d]" 

def get_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--download', choices = ['dataset', 'str_exp', 'str', 'connect_exp', 'connect_dat'], required=True, type=str, help = 'String of SNP ID field')
	args = parser.parse_args()
	return args

def download_All_Mouse_Brain_ISH_experiments():
	url = """http://api.brain-map.org/api/v2/data/query.csv?criteria=model::SectionDataSet,rma::criteria,[failed$eqfalse],products[abbreviation$eq'Mouse'],treatments[name$eq'ISH'],genes,plane_of_section,rma::options,[tabular$eq'plane_of_sections.name+as+plane','genes.acronym+as+gene','data_sets.id+as+section_data_set_id'],[order$eq'plane_of_sections.name,genes.acronym,data_sets.id']&start_row=0&num_rows=all"""
	print(url)
	r = requests.get(url, allow_redirects=True)
	open('All_Mouse_Brain_ISH_experiments.csv', 'wb').write(r.content)

def download_sections_All_Mouse_Brain_ISH_experiments():
	url_base = "%s/StructureUnionize/query.csv" +\
			"?criteria=[section_data_set_id$eq%d]" +\
			(",structure[graph_id$eq%d]" % GRAPH_ID) +\
			"&numRows=all"
	reader = csv.reader(open("All_Mouse_Brain_ISH_experiments.csv", 'rt'), delimiter=",")
	head = next(reader)
	for i, row in enumerate(reader):
		section_id = int(row[2])
		print(i, section_id)
		url = url_base % (API_PATH, section_id)
		r = requests.get(url, allow_redirects=True)
		open('/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/%d.csv'%section_id, 'wb').write(r.content)

def download_structures():
	url = STRUCTURES_URL #% (API_PATH, GRAPH_ID)
	r = requests.get(url, allow_redirects=True)
	open('./structures.csv', 'wb').write(r.content)

# Download structure-unionized projection data
def download_connectivity_exp():
	mcc = MouseConnectivityCache(manifest_file='dat/mouse_connectivity_manifest.json')
	all_experiments = mcc.get_experiments(dataframe=True)
	print("%d total experiments" % len(all_experiments))
	all_experiments.to_csv("dat/mouse_connectivity_all_experiments.csv", index=False)

def download_connectivity_dat():
	mouse_conn_exp = pd.read_csv("/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/src/dat/mouse_connectivity_experiments.csv")
	for i, row in mouse_conn_exp.iterrows():
		_id = row["id"]
		print(i, _id)
		url = STRUCTURE_UNIONIZE_CONN_URL%(API_PATH, _id)
		r = requests.get(url, allow_redirects=True)
		open('/ifs/scratch/c2b2/dv_lab/jw3514/circuits-jw/dat/allen-mouse-conn/raw/%d.csv'%_id, 'wb').write(r.content)


def main():
	args = get_options()
	if args.download == "dataset":
		print("Download ISH section dataset info")
		download_All_Mouse_Brain_ISH_experiments()
	elif args.download == "str_exp":
		print("Download UNIONIZED ISH expression data")
		download_sections_All_Mouse_Brain_ISH_experiments()
	elif args.download == "str":
		print("Download Structures info")
		download_structures()
	elif args.download == "connect_exp":
		print("Download UNIONIZED CONNECTIVITY DATA INFO")
		download_connectivity_exp()
	elif args.download == "connect_dat":
		print("Download UNIONIZED CONNECTIVITY DATA SET")
		download_connectivity_dat()

if __name__== "__main__":
	main()

