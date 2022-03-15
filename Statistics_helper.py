# Draft of statistical comparisons
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
def compare_layer(l1, l2):
    w1 = l1.weight
    w2 = l2.weight
    MAE = sum(abs(sum(w1 - w2)))
    MSE = sum((w1 - w2) ** 2) / len(w1)
    return MAE, MSE


def make_pca_agg_fit(
    seed,
    data,
    variability,
    comp_guas,
    func_give=KMeans,
    array_out=False,
    loud=False,
):
    flag = True
    start = 2
    while flag:
        Out = PCA(n_components=start,random_state=seed)
        g = Out.fit(data)
        check = sum(Out.explained_variance_ratio_)
        hold = Out.explained_variance_ratio_
        if check > variability:
            flag = False
        if start > 10:
            flag = False
        start += 1
    if start > 2 and loud:
        print(
            f"There were {start} components with net variance explained {check} (max iter = 100), this only support plotting 2"
        )
    a = g.transform(data)
    pc1 = a.T[:][0]
    pc2 = a.T[:][1]
    if loud:
        plt.scatter(pc1, pc2)
    func = func_give(n_clusters=comp_guas,random_state=seed)
    color = func.fit_predict(data)
    if loud:
        plt.scatter(pc1, pc2, c=color)
        plt.title("PC 1 vs PC 2")
        plt.xlabel(f"PC 1 (var = {int(hold[0]*100)}%)")
        plt.ylabel(f"PC 2 (var = {int(hold[1]*100)}%)")
    if array_out:
        return pc1, pc2, color


def make_pca_gausian_fit(seed,data, variability, comp_guas, array_out=False, loud=False):
    flag = True
    start = 2
    while flag:
        Out = PCA(n_components=start,random_state=seed)
        g = Out.fit(data)
        check = sum(Out.explained_variance_ratio_)
        hold = Out.explained_variance_ratio_
        if check > variability:
            flag = False
        if start > 10:
            flag = False
        print(start)
        start += 1
    if start > 2 and loud:
        print(
            f"There were {start} components with net variance explained {check}, this only support plotting 2"
        )
    a = g.transform(data)
    pc1 = a.T[:][0]
    pc2 = a.T[:][1]
    if loud:
        plt.scatter(pc1, pc2)
    Gaus = GaussianMixture(n_components=comp_guas,random_state=seed)
    Gaus = Gaus.fit(data)
    color = Gaus.predict(data)
    if loud:
        plt.scatter(pc1, pc2, c=color)
        plt.title("PC 1 vs PC 2")
        plt.xlabel(f"PC 1 (var = {int(hold[0]*100)}%)")
        plt.ylabel(f"PC 2 (var = {int(hold[1]*100)}%)")
    if array_out:
        return pc1, pc2, color


def time_series_with_error(hold, compare, interest, index):
    series = []
    labels = []
    for i in hold[compare].unique():
        series.append(hold[((hold[compare] == i))])
        labels.append(i)
    fig, ax = plt.subplots()
    for i in range(len(series)):
        obj = series[i].groupby(index).mean()[interest][1:]
        std = series[i].groupby(index).std()[interest][1:]
        upper = [min(1, a) for a in obj + std]
        Lower = [max(0, a) for a in obj - std]
        obj = [max(0, a) for a in obj]
        index_1 = sorted(series[0]["Iteration"].unique())[1:]
        ax.fill_between(index_1, upper, Lower, alpha=0.2, linewidth=0)
        ax.plot(index_1, obj, label=labels[i])
        ax.legend()
def rescale(test):
    g=preprocessing.StandardScaler().fit_transform(test)
    g=pd.DataFrame(g)
    g.columns=test.columns
    return g


def sample_cluster_frame(seed,frame,clusters,predict,sample=100,output=False):
    assert sample % len(np.unique((clusters))) == 0
    frame["Cluster"]=clusters
    Y=frame[str(predict)]
    PFE=Y
    frame.drop([predict],axis=1)
    X_train=pd.DataFrame([])
    X_test=pd.DataFrame([])
    y_train=pd.DataFrame([])
    y_test=pd.DataFrame([])
    if output:
        Cluster_df=[]
    for count,i in enumerate(sorted(frame["Cluster"].unique())):
        frame_t=frame[frame["Cluster"]==i]
        sampled_frame=frame_t.sample(int(sample/len(np.unique(clusters))),random_state=seed,replace=True)
        Y=PFE.iloc[sampled_frame.index.values]
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(sampled_frame, Y, test_size=0.2, random_state=seed)
        X_train=X_train.append(X_train_1)
        X_test=X_test.append(X_test_1)
        y_train=pd.concat([y_train,y_train_1.transpose()])
        y_test=pd.concat([y_test,y_test_1.transpose()])
        if output:
            Cluster_df.append(frame)
    y_train=y_train.rename(columns={0:predict})
    y_test=y_test.rename(columns={0:predict})
    if output:
        return Cluster_df,X_train,X_test,y_train,y_test
    return X_train,X_test,y_train,y_test 

def stratified_cluster_sample(seed,frame,feature_array,interest,n_cluster,var_ratio=.01,sample=100,C_type=KMeans,net_out=False):
    feat_5=frame[feature_array]
    r_feat_5=rescale(feat_5)
    pc1,pc2,color=make_pca_agg_fit(seed,r_feat_5,var_ratio,n_cluster,func_give=C_type,array_out=True,loud=False)
    r_feat_5[interest]=frame[interest]
    if net_out:
        df,X_train,X_test,y_train,y_test=sample_cluster_frame(seed,r_feat_5,color,interest,sample,output=True)
        return df,X_train,X_test,y_train,y_test
    else:
        X_train,X_test,y_train,y_test=sample_cluster_frame(seed,r_feat_5,color,interest,sample,output=False)
        return X_train,X_test,y_train,y_test