#DIR="/home/jw3514/Work/ASD_Circuits/scripts/"
#InpFil=$DIR"RankScores/RankScore.Ipsi.Cont.npy"
#Prefix="SI.All"
#nohup bash run_Get_P_Mat.sh $InpFil $Prefix & 

#InpFil=$DIR"RankScores/RankScore.Ipsi.Short.3900.Cont.npy"
#Prefix="SI.Short"
#nohup bash run_Get_P_Mat.sh $InpFil $Prefix &

#InpFil=$DIR"RankScores/RankScore.Ipsi.Long.3900.Cont.npy"
#Prefix="SI.Long"
#nohup bash run_Get_P_Mat.sh $InpFil $Prefix &

DIR="/home/jw3514/Work/ASD_Circuits/scripts/"
InpFil=$DIR"RankScores/RankConn.C.Ipsi.Cont.npy"
Prefix="Conn.All"
nohup bash run_Get_P_Mat.sh $InpFil $Prefix & 

InpFil=$DIR"RankScores/RankConn.C.Ipsi.Short.3900.Cont.npy"
Prefix="Conn.Short"
nohup bash run_Get_P_Mat.sh $InpFil $Prefix &

InpFil=$DIR"RankScores/RankConn.C.Ipsi.Long.3900.Cont.npy"
Prefix="Conn.Long"
nohup bash run_Get_P_Mat.sh $InpFil $Prefix &
