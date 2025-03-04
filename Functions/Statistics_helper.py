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
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram
from .helper import PCA_order_swap
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
    """
    Makes a clustering fit to pca data from raw data provided 

    Inputs
    ------
    seed (int) - random seed
    data (df or numpy array) - raw data to fit
    variability (0-1 real) - how much variability you want in your comp. 
    comp_guas (int) - k means cluster to use
    func_give (sklearn function) - what type of clustering (kmean default)
    array_out (bool) - if return data
    loud (bool) - if print graphs of pca

    Outputs
    ------
    pc1 (array) - pc1 components
    pc2 (array - pc2 components
    color (array) - cluster tags
    """
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
    """
    Makes a clustering fit to pca data from raw data provided using specifically guassian clustering

    Inputs
    ------
    seed (int) - random seed
    data (df or numpy array) - raw data to fit
    variability (0-1 real) - how much variability you want in your comp. 
    comp_guas (int) - k means cluster to use
    array_out (bool) - if return data
    loud (bool) - if print graphs of pca

    Outputs
    ------
    pc1 (array) - pc1 components
    pc2 (array - pc2 components
    color (array) - cluster tags
    """
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
    """
    plots a time series with corresponding error 

    Inputs
    ------
    hold (df) - df of data to prepare (form of df has a interested column and a type of interest : ie group by activation function then look at test set)
    compare (array of str) - subset of data frame
    interest (str) - column of interest to analyize
    index (str) - what to group by (see hold)

    Outputs
    ------
     * displays plot, no object return *
    """
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
    """
    Standard scales data

    Inputs
    ------
    test (array or df) - data to transform

    Outputs
    ------
    g (df) - rescaled frame
    """
    g=preprocessing.StandardScaler().fit_transform(test)
    g=pd.DataFrame(g)
    g.columns=test.columns
    return g


def sample_cluster_frame(seed,frame,clusters,predict,sample=100,output=False):
    """
    stratified samples from frame with respect to the clusters

    Inputs
    ------
    seed(int) -random seed
    frame (df) - dataset frame with clusters already appened and refered to as "Cluster" column
    clusters (int) - # refering to x row (ie [0,1,...] row 0 - cluster 0, row 1 - cluster 1,etc)
    predict (df / array) - dependent variable to predict
    sample (int) - nunber to sample per cluster

    Outputs
    ------
    X_train (df) - var train set 
    X_test (df) - var test set 
    y_train (df) - prediction train set 
    y_test  (df) - prediciton test set 
    """
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
    """
    pairs above functions together to go from raw to stratified sample

    Inputs
    ------
    seed(int) -random seed
    frame (df) - dataset frame with clusters already appened and refered to as "Cluster" column
    feature array (array of str) - col names of features in frame
    var_ratio (0-1 real) - ratio needed for clustering
    interest (str) - prediction value to use (dependent variable)
    n_clusters (int) - number of clusters to use
    c_type (func) - function used to cluster
    net_out (bol) - returns test / train splits with df, if no, df is not returned

    Outputs
    ------
    X_train (df) - var train set 
    X_test (df) - var test set 
    y_train (df) - prediction train set 
    y_test  (df) - prediciton test set 
    if net out:
        df (df) - net dataframe
    """
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

def add_pca_and_graph(data2,pc1,pc2,color,graph=True,loud=True):
    """
    plots pca space with clusters and labels

    Inputs
    ------
    data2 (df) - df of data (index matches other inputs)
    pc1 (array) -array of pc1 data (index matches other inputs)
    pc2 (array) - array of pc2 data (index matches other inputs)
    color (array) - array of cluster labels as int

    Outputs
    ------
    data2 (df) - data2 returned with the addition of pc1,pc2,cluster

    """
    dic={
    "Pc1" : pc1,
    "Pc2" : pc2,
    "Cluster" : color,
    }
    holder=pd.DataFrame(dic)
    data2=pd.concat([data2,holder],axis=1)
    data2["Cluster"]=data2["Cluster"].apply(PCA_order_swap)
    if graph:
        plt.scatter(data2["Pc1"],data2["Pc2"],c=data2["Cluster"])
        plt.ylabel("Pc2")
        plt.xlabel("Pc1")
        plt.title("PC Based Clustering")
        #note made for 6 clusters currently
        annotations=["C0","C1","C2","C3","C4","C5"]
        new=data2[['Pc1', 'Pc2', 'Cluster']].groupby("Cluster").mean()
        for i, label in enumerate(annotations):
            plt.annotate(label, (new["Pc1"][i],new["Pc2"][i]),weight='bold',c="r",size=12)
    return data2

def plot_outline(abridge):
    """
    uses convex hull and manual functions to plot lines representing pca space of MOFs

    Inputs
    ------
    abridge (df with col = "Pc1","Pc2") - data to plot

    Outputs
    ------
    **plots outline **

    """
    alpha_tuples=[[a,b] for a,b in zip(abridge["Pc1"].to_numpy(),abridge["Pc2"].to_numpy())]
    alpha_tuples=np.array(alpha_tuples)
    hull=ConvexHull(alpha_tuples)
    x=np.linspace(-4.2,-1.6,1001)
    y= lambda x: -.5 - 1.5*x
    plt.plot(x,y(x),'r--', lw=2)
    x=np.linspace(-5,-4.2,1001)
    y= lambda x: 2.6- .75*x
    plt.plot(x,y(x),'r--', lw=2)
    x=np.linspace(-1.6,5.5,1001)
    y= lambda x: 2 + .1*x
    plt.plot(x,y(x),'r--', lw=2)
    x=np.linspace(-1.6,5,1001)
    y= lambda x: 2 + .1*x
    plt.plot(x,y(x),'r--', lw=2)
    plt.plot(alpha_tuples[hull.vertices,0][:10], alpha_tuples[hull.vertices,1][:10], 'r--', lw=2)
    plt.show()

def plot_dendrogram(model,a, **kwargs):
    """
    plots dendrogram 
    - see https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html for more 
    - b/c of **kwargs, see for reference on what is possible to pass

    Inputs
    ------
    model(object) - model of type generated by agg.  clustering 
    a (arr of str) - labels for classes

    Outputs
    ------
    **plots dendrogram **

    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,orientation='left',labels=a.index,**kwargs)

def size_clusters(hold):
    """
    resize cluster to be teh same size

    Inputs
    ------
    hold (array) - array or df of cluster size

    Outputs
    ------
    new (array) - sampled array of where each is sampled where len() =  len of smallest sample in hold

    """
    temp=[]
    new=[]
    for i in hold:
        temp.append(len(i))
    size=min(temp)
    for i in hold:
        new.append(i.sample(size))
    return new