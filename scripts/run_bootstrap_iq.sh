Start=1
Step=100
End=1000
Script="script_bootstrapping_mutations.py"

#DIR_HIQ="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.HIQ/"
#DIR_LIQ="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.LIQ/"
#mkdir $DIR_HIQ $DIR_LIQ
#for i in $(seq $Start $Step $End);
#do
#	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Pheno.HighIQ.Highconf.Muts.csv -l $DIR_HIQ > boot.hiq.log &
#	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Pheno.LowIQ.Highconf.Muts.csv -l $DIR_LIQ > boot.liq.log &
#done


#DIR_Male="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.Male/"
#DIR_Female="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.Female/"
#mkdir $DIR_Male $DIR_Female
#for i in $(seq $Start $Step $End);
#do
#	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Gender.Male.Highconf.Muts.csv -l $DIR_Male > boot.male.log &
#	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Gender.Female.Highconf.Muts.csv -l $DIR_Female > boot.female.log &
#done

DIR_HIQ_Male="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.HIQ.Male/"
DIR_HIQ_Female="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.HIQ.Female/"
DIR_LIQ_Male="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.LIQ.Male/"
DIR_LIQ_Female="/home/jw3514/Work/ASD_Circuits/dat/Unionize_bias/Bootstrap/ASD.LIQ.Female/"
mkdir $DIR_HIQ_Male $DIR_HIQ_Female $DIR_LIQ_Male $DIR_LIQ_Female
for i in $(seq $Start $Step $End);
do
	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Pheno.HighIQ.Male.Highconf.Muts.csv -l $DIR_HIQ_Male > boot.hiq.male.log &
	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Pheno.HighIQ.Female.Highconf.Muts.csv -l $DIR_HIQ_Female > boot.hiq.female.log &
	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Pheno.LowIQ.Male.Highconf.Muts.csv -l $DIR_LIQ_Male > boot.liq.male.log &
	nohup python $Script --nstart $i --nend $(($i+$Step-1)) -v ../dat/Unionize_bias/Pheno.LowIQ.Female.Highconf.Muts.csv -l $DIR_LIQ_Female > boot.liq.female.log &
done
