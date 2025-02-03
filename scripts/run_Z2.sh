#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -N calculate_Z2
#$ -l h_rt=24:00:00
#$ -l h_vmem=1G
#$ -cwd
Script="script_Z2_calculation.py"
#python $Script --start $1 --step $2 -i $3
python $Script --start $1 --step $2 
