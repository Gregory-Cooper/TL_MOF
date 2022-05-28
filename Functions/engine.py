from Functions.dataset_loader import data_loader, get_descriptors, one_filter, data_scaler
import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from .Statistics_helper import stratified_cluster_sample
import itertools
import scipy
import pandas as pd
from scipy.spatial import distance_matrix

def get_processed_data(unprocessed=False):
    """
    gets processed data for use in clustering primarily
    """
    # file name and data pathv 
    base_path = os.getcwd()
    file_name = 'data/CrystGrowthDesign_SI.csv'
    # load data
    data = data_loader(base_path, file_name)
    data = data.reset_index(drop=True)
    if unprocessed:
        return data
    # prepare training inputs and outputs
    test=data.drop(["MOF ID","topology","First nodular character","Second nodular character"],axis=1)
    test=test[['void fraction', 'Vol. S.A.', 'Grav. S.A.', 'Pore diameter Limiting', 'Pore diameter Largest']]
    #processes
    g=preprocessing.StandardScaler().fit_transform(test)
    g=pd.DataFrame(g)
    g.columns=test.columns
    return g

def prep_data_splits(data,descriptor_columns,one_filter_columns,test_size=.2):
    df, t_1, t_2, y_1, y_2 = stratified_cluster_sample(
        1, data, descriptor_columns, one_filter_columns[0], 5, net_out=True
    )
    df = df[0]
    df=df.drop("Cluster",axis=1)
    #descriptor_columns.append("Cluster")
    interest = one_filter_columns[0]
    #descriptor_columns.append("Cluster")
    features = descriptor_columns

    df_train, df_test, y_df_train, y_df_test = train_test_split(
        df[features], df[interest], test_size=test_size
    )
    y_df_train=y_df_train.reset_index(drop=False)
    df_train, df_val, y_df_train, y_df_val = train_test_split(
        df_train[features], y_df_train[interest], test_size=test_size
    )
    df_train[interest] = np.array(y_df_train)
    df_val[interest] = np.array(y_df_val)
    df_test[interest]=np.array(y_df_test)
    return df_train,df_val,df_test

def create_dictionaries(size):
    indexer=itertools.product(range(size),range(size))
    dic={}
    for i in indexer:
        dic[i]=[]
    indexer=itertools.product(range(size),range(size))
    dic2={}
    for i in indexer:
        dic2[i]=[]
    std_store={}
    for i in indexer:
        std_store[i]=[]
    return dic,dic2,std_store

def unpack_dic(dic,meta):
    for i in meta:
        for g in i:
            for count,z in enumerate(i[g]):
                dic[(g,count)].append(z)
    return dic

def create_std_matrix(dic,std_store):
    for z in dic:
        matrix=np.matrix(dic[z][0])
        for count,i in enumerate(dic[z]):
            if count==0:
                pass
            else:
                i=np.matrix(i)
                matrix=np.concatenate([matrix,i],axis=0)
        matrix=np.matrix(matrix)
        std_store[z]=np.array(matrix.std(axis=0,dtype=float))[0]
    return std_store

def preformance_graph_and_prep_2nd_set(Cluster_colors,dic,dic2,adjust,save=False):
    last=0
    Up_max=[]
    for i in dic:
        f_index=i[0]
        if f_index is not last:
            for count,z in enumerate(Up_max):
                plt.errorbar(count,z.mean(),yerr=z.std(),fmt="o",c=Cluster_colors[count])
            plt.title(f"Final Preformance for Base Cluster {f_index-1}")
            plt.ylabel("R^2 Preformance on Test set")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Final_preformance{i}.png",dpi=400)
            plt.show()
            Up_max=[]
        store_max=[]
        store_all=[]
        for count,g in enumerate(dic[i]):
            if count==0:
                store_all=np.array(g)
            else:
                store_all=store_all+np.array(g)
            store_max.append(max(g))
        store_max=np.array(store_max)
        store_all=np.array(store_all)
        dic2[i].append((store_all/adjust))
        Up_max.append(store_max)
        last=f_index
    for count,z in enumerate(Up_max):
        plt.errorbar(count,z.mean(),yerr=z.std(),fmt="o",c=Cluster_colors[count])
    plt.title(f"Final Preformance for Base Cluster {5}")
    plt.ylabel("R^2 Preformance on Test set")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Final_preformance{i}.png",dpi=400)
    plt.show()
    return dic2

def Transfer_graphs(dic2,resolution,epoch_conversions,Cluster_colors,byte,std_store,epochs,save=False):
    dif_holder=[]
    std_diff=[]
    overfit_holder=[]
    residual_holder=[]
    std_res=[]
    integral_holder=[]
    std_int=[]
    error_holder=[]
    last=0
    difference=.1
    GB=True
    for i in dic2:
        f_index=i[0]
        #integral and residual 
        y=(max(dic2[i][0]))-dic2[i][0]
        x=np.linspace(1,epochs,resolution)
        index=np.where(y<(difference))
        ### overfit
        #std_test
        sy=(max(dic2[i][0]+std_store[i]))-dic2[i][0]+std_store[i]
        sx=np.linspace(1,epochs,resolution)
        sindex=np.where(sy<(difference))
        #regular diffs
        diff=max(dic2[i][0])-min(dic2[i][0])
        std1=std_store[i][0]
        #overfit test
        overfit=(dic2[i][0][0]-dic2[i][0][-1]) > 0
        if f_index is not last:
            for count,z in enumerate(dif_holder):
                if overfit_holder[count]:
                    plt.errorbar(count,-z,marker="x",c=Cluster_colors[count],yerr=std_diff[count])
                else:
                    plt.errorbar(count,z,marker="o",c=Cluster_colors[count],yerr=std_diff[count])
            plt.title(f"Average Training Deviation on the Worst Epoch for Base {i[0]-1}")
            plt.ylabel("R^2 deviation from best preforming model")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Training_deviation{i[0]-1}.png",dpi=400)
            plt.show()
            for count,g in enumerate(residual_holder):
                if g == "error":
                    plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
                else:
                    conversion=epoch_conversions[count]
                    g=g*epoch_conversions[count]
                    plt.ylabel("Datapoints")
                    if GB:
                        conversion=conversion*byte*0.000001
                        g=g*byte*0.000001 #mega bytes
                        plt.ylabel("MegaBytes")
                    plt.errorbar(count,g,c=Cluster_colors[count],yerr=std_res[count]*conversion,fmt="o")
            plt.title(f"Average Information to reach {difference*100} % Deviation Base {i[0]-1}")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Average_info{i[0]-1}.png",dpi=400)
            plt.show()
            for count,f in enumerate(integral_holder):
                if f == "error":
                    plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
                else:
                    plt.errorbar(count,f,c=Cluster_colors[count],yerr=std_int[count],fmt="o")
            plt.title(f"Learning Efficency to reach {difference} R^2 Deviation : Base {i[0]-1}")
            plt.ylabel("Net R^2 deviation squared")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Integral{i[0]-1}.png",dpi=400)
            plt.show()
            dif_holder=[]
            std_diff=[]
            overfit_holder=[]
            residual_holder=[]
            std_res=[]
            integral_holder=[]
            std_int=[]
        dif_holder.append(diff)
        std_diff.append(std1)
        overfit_holder.append(overfit)
        try:
            first=index[0][0]
            sfirst=index[0][0]
            integral=scipy.integrate.simps(y[:first], x=x[:first], dx=1, axis=-1, even='first')
            sintegral=scipy.integrate.simps(sy[:sfirst], x=sx[:sfirst], dx=1, axis=-1, even='first')
            integral_holder.append(integral)
            std_int.append(sintegral-integral)
            print(integral,sintegral,sx[sfirst],x[first])
            residual_holder.append(x[first])
            std_res.append(sx[sfirst]-x[first])
        except:
            print("error", i)
            integral_holder.append("error")
            std_int.append(0)
            residual_holder.append("error")
            std_res.append(0)
        last=f_index
    for count,z in enumerate(dif_holder):
        if overfit_holder[count]:
            plt.errorbar(count,-z,marker="x",c=Cluster_colors[count],yerr=std_diff[count])
        else:
            plt.errorbar(count,z,marker="o",c=Cluster_colors[count],yerr=std_diff[count])
    plt.title(f"Average Training Deviation on the Worst Epoch for Base {5}")
    plt.ylabel("R^2 deviation from best preforming model")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Training_Deviation{5}.png",dpi=400)
    plt.show()
    for count,g in enumerate(residual_holder):
        if g == "error":
            plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
        else:
            conversion=epoch_conversions[count]
            g=g*epoch_conversions[count]
            plt.ylabel("Datapoints")
            if GB:
                conversion=conversion*byte*0.000001
                g=g*byte*0.000001 #mega bytes
                plt.ylabel("MegaBytes")
            plt.errorbar(count,g,c=Cluster_colors[count],yerr=std_res[count]*conversion,fmt="o")
    plt.title(f"Average Information to reach {difference*100} % Deviation Base {5}")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Information{5}.png",dpi=400)
    plt.show()
    for count,f in enumerate(integral_holder):
        if f == "error":
            plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
        else:
            plt.errorbar(count,f,c=Cluster_colors[count],yerr=std_int[count],fmt="o")
    plt.title(f"Average Training time to reach {difference*100} % Deviation Base {5} ")
    plt.ylabel("Net R^2 deviation squared")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Training_Time{5}.png",dpi=400)
    plt.show()

def make_distance_graph(dic,distances,Cluster_colors,name,save=False):
    last=0
    Up_max=[]
    for i in dic:
        f_index=i[0]
        if f_index is not last:
            for count,z in enumerate(Up_max):
                temp=distances[f_index-1]
                plt.errorbar(temp[count],z.mean(),yerr=z.std(),fmt="o",c=Cluster_colors[count],label=f"Cluster {count}")
            plt.title(f"Final Preformance for Base Cluster {f_index-1}")
            plt.ylabel("R^2 Preformance on Test set")
            plt.xlabel(f"{name} Distance")
            plt.legend()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if save:
                plt.savefig(f"Distance_Graph_{f_index-1}.png")
            plt.show()
            Up_max=[]
        store_max=[]
        store_all=[]
        for count,g in enumerate(dic[i]):
            if count==0:
                store_all=np.array(g)
            else:
                store_all=store_all+np.array(g)
            store_max.append(max(g))
        store_max=np.array(store_max)
        store_all=np.array(store_all)
        #dic2[i].append((store_all/adjust))
        Up_max.append(store_max)
        last=f_index
    for count,z in enumerate(Up_max):
        temp=distances[len(distances)-1]
        plt.errorbar(temp[count],z.mean(),yerr=z.std(),fmt="o",c=Cluster_colors[count],label=f"Cluster {count}")
    plt.title(f"Final Preformance for Base Cluster {5}")
    plt.ylabel("R^2 Preformance on Test set")
    plt.xlabel(f"{name} Distance")
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(f"Distance_Graph_{5}.png")
    plt.show()