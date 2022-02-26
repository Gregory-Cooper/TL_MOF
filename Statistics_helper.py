# Draft of statistical comparisons
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
def compare_layer(l1,l2):
    w1=l1.weight
    w2=l2.weight
    MAE=sum(abs(sum(w1-w2)))
    MSE= sum((w1-w2)**2)/len(w1)
    return MAE,MSE

def make_pca_agg_fit(data,variability,comp_guas,func_give=AgglomerativeClustering,array_out=False):
    flag=True
    start=2
    while flag:
        Out=PCA(n_components=start)
        g=Out.fit(data)
        check=sum(Out.explained_variance_ratio_)
        hold=Out.explained_variance_ratio_
        if check > variability:
            flag=False
        if start > 10:
            flag=False
        print(start)
        start+=1
    if start > 2:
        print(f"There were {start} components with net varance explained {check} (max iter = 100), this only support plotting 2")
    a=g.transform(data)
    pc1=a.T[:][0]
    pc2=a.T[:][1]
    plt.scatter(pc1,pc2)
    func=func_give(n_clusters=comp_guas)
    color=func.fit_predict(data)
    plt.scatter(pc1,pc2,c=color)
    plt.title("PC 1 vs PC 2")
    plt.xlabel(f"PC 1 (var = {int(hold[0]*100)}%)")
    plt.ylabel(f"PC 2 (var = {int(hold[1]*100)}%)")
    if array_out:
        return a,color

def make_pca_gausian_fit(data,variability,comp_guas,array_out=False):
    flag=True
    start=2
    while flag:
        Out=PCA(n_components=start)
        g=Out.fit(data)
        check=sum(Out.explained_variance_ratio_)
        hold=Out.explained_variance_ratio_
        if check > variability:
            flag=False
        if start > 10:
            flag=False
        print(start)
        start+=1
    if start > 2:
        print(f"There were {start} components with net varance explained {check}, this only support plotting 2")
    a=g.transform(data)
    pc1=a.T[:][0]
    pc2=a.T[:][1]
    plt.scatter(pc1,pc2)
    Gaus=GaussianMixture(n_components=comp_guas)
    Gaus=Gaus.fit(data)
    color=Gaus.predict(data)
    plt.scatter(pc1,pc2,c=color)
    plt.title("PC 1 vs PC 2")
    plt.xlabel(f"PC 1 (var = {int(hold[0]*100)}%)")
    plt.ylabel(f"PC 2 (var = {int(hold[1]*100)}%)")
    if array_out:
        return a,color