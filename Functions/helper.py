
def PCA_order_swap(x):
    #swaps clusters to order from left to right on pca
    #probably should have useed some type of map
    x=int(x)
    y=0
    if x == 5:
         y=0
    elif x == 2:
        y= 1
    elif x== 1:
        y=2
    elif x == 4:
        y=3
    elif x == 0:
        y=4
    else:
        y=5
    return y

def count_clusters(abridge):
    """
    counts how many data points are in each cluster per topology, uses a data frame
    """
    dic={}
    for i in abridge["topology"].unique():
        dic[i]=[]
    for i in range(6):
        Cluster1=abridge[abridge["Cluster"]==i]
        for j in abridge["topology"].unique():
            if sum(Cluster1["topology"]==j) > 0:
                dic[j].append(sum(Cluster1["topology"]==j))
            else:
                dic[j].append(0)
    return dic