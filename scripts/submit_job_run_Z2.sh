#!/bin/bash
Start=0
Step=1000
End=20000
Script="script_Z2_calculation.py"
#MAT=dat/ExpMat/16Regions.mouse.Zscore.csv
#MAT="JW_Z1-Mat.ArithmeticMean.0422.csv"
#MAT="/home/jw3514/Work/ASD_Circuits/dat/Human.CT.Z1.csv"
MAT="JW_ExpMat.LogMean.0418.csv"

parallel -j 20 bash run_Z2.sh {} $Step $MAT ::: $(seq $Start $Step $End)
#bash run_Z2.sh 0 $Step $MAT 
