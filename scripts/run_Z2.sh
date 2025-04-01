#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -N calculate_Z2
#$ -l h_rt=24:00:00
#$ -l h_vmem=1G
#$ -cwd
Script="script_Z2_calculation.py"
#Script="script_Spec2_calculation.py"
#echo "python $Script --start $1 --step $2 -i $3, -m $4 -o $5"
python $Script --start $1 --step $2 -i $3 -m $4 -o $5
