#INPUT="../dat/genes/Jon.ssc.txt"
#OUTPUT="dat/bias/asd.ssc.zmatch.rank.csv"
#MATCH="dat/asd_ssc_exp_matches_1000.csv"

#INPUT="../dat/genes/Jon.spark.txt"
#OUTPUT="dat/bias/asd.spark.zmatch.rank.csv"
#MATCH="dat/asd_spark_exp_matches_1000.csv"

#INPUT="../dat/genes/Jon.tada.txt"
#OUTPUT="dat/bias/asd.tada.zmatch.rank.csv"
#MATCH="dat/asd_tada_exp_matches_1000.csv"
#python match_genes.py -i $INPUT -o $OUTPUT -m $MATCH

#INPUT="../dat/genes/asd.sib.entrez.list"
#OUTPUT="dat/bias/asd.sib.zmatch.rank.csv"
#MATCH="dat/asd_sib_exp_matches_1000.csv"
#python match_genes.py -i $INPUT -o $OUTPUT -m $MATCH

INPUT="../dat/genes/asd.asc.entrez.list"
OUTPUT="dat/bias/asd.asc.zmatch.rank.csv"
MATCH="dat/asd_asc_exp_matches_1000.csv"
python match_genes.py -i $INPUT -o $OUTPUT -m $MATCH
