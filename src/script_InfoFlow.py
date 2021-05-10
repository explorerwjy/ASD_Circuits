from ASD_Circuits import *

def InfoFlow(weight_flag = True):
    graph = LoadConnectome2(ConnFil="../dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight.csv")
    STR_Labels = []
    for idx in range(len(graph.vs)):
        STR_Labels.append(graph.vs[idx]["label"])

    EdgeDict = {}
    for idx in range(len(graph.vs)):
        keys = []
        weights = []
        for edge in graph.es.select(_source=idx):
            #print(edge.source, edge.target, edge["weight"])
            keys.append("{}-{}".format(edge.source, edge.target))
            weights.append(edge["weight"])
        weights = np.array(weights)
        weights = weights/np.sum(weights)
        weights[-1] = 1 - np.sum(weights[0:-1])
        for key, weight in zip(keys, weights):
            EdgeDict[key] = weight

    NTrials = 100
    Nstep = 1000
    Dat = {}
    for idx in range(len(graph.vs)):
        print(idx, graph.vs[idx]["label"])
        Vertex = graph.vs[idx]
        Visited = []
        for i in range(NTrials):
            next_ = Vertex
            for j in range(Nstep):
                successors = next_.successors()
                if weight_flag:
                    Probs = []
                    for suc in successors:
                        bs = EdgeDict["{}-{}".format(next_.index, suc.index)]
                        Probs.append(bs)
                    next_ = np.random.choice(successors, p=Probs)
                else:
                    next_ = np.random.choice(successors)
                Visited.append(next_["label"])
        Dat[Vertex] = CountVisitFreq(Visited, STR_Labels)

    df_dat = []
    for idx in range(len(graph.vs)):
        Vertex = graph.vs[idx]
        df_dat.append(Dat[Vertex])

    df = pd.DataFrame(data=df_dat, index=STR_Labels)
    df.to_csv("dat/Conn.Nonweight2.csv")

def CountVisitFreq(List, STR_Labels):
    tmp_dict = {}
    for STR in STR_Labels:
        tmp_dict[STR] = 0
    for x in List:
        tmp_dict[x] += 1
    return tmp_dict

InfoFlow(weight_flag = False)
