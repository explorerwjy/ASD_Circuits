import sys
import os
import argparse
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src/')
from CellType_PSY import *


def combine_z2(input_dir, output_file):
    DFs = []
    for file in os.listdir(input_dir):
        df = pd.read_csv(os.path.join(input_dir, file), index_col=0)
        DFs.append(df)
    Z2_Mat = pd.concat(DFs)
    Z2_Mat.to_csv(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Z2 split files into a single matrix.")
    parser.add_argument("input_dir", help="Directory containing Z2 split files")
    parser.add_argument("output_file", help="Path to save the combined Z2 matrix")  # Add default name as input_dir name 
    args = parser.parse_args()

    combine_z2(args.input_dir, args.output_file)