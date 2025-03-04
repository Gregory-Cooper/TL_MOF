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

    Outputs 
    -------
    g (df) - processed data in form for analysis
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
    """
    A function to prepare the training, validation, and test data splits.

    Args:
    - data: A pandas dataframe that includes the data.
    - descriptor_columns: A list of strings with column names that contain the features of interest.
    - one_filter_columns: A list of strings with column names that include the filter values.
    - test_size: A float that represents the ratio of the test set size to the size of the entire dataset.

    Returns:
    - df_train: A pandas dataframe that includes the training data for the model.
    - df_val: A pandas dataframe that includes the validation data for the model.
    - df_test: A pandas dataframe that includes the test data for the model.
    """
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
    """
    A function to create dictionaries with keys that are tuples of size 2.

    Args:
    - size: An integer that represents the size of the tuples.

    Returns:
    - dic: A dictionary that has tuple keys and empty list values.
    - dic2: A dictionary that has tuple keys and empty list values.
    - std_store: A dictionary that has tuple keys and empty list values.
    """
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
    """
    A function to unpack the contents of a dictionary into a low level dictionary.

    Args:
    - dic: A dictionary with tuple keys and list values.
    - meta: A list of lists with tuple keys that are in the given dictionary.

    Returns:
    - dic: A dictionary that has tuple keys and list values.
    """
    for i in meta:
        for g in i:
            for count,z in enumerate(i[g]):
                dic[(int(g),count)].append(z)
    return dic

def create_std_matrix(dic,std_store):
    """
    A function to create a standard deviation matrix for the given dictionary.

    Args:
    - dic: A dictionary with tuple keys and list values.
    - std_store: A dictionary with tuple keys and empty list values.

    Returns:
    - std_store: A dictionary that has tuple keys and list values of the standard deviation matrix.
    """
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
    """
    A function to prepare the second set of data for the model and plot the performance graph.

    Args:
    - Cluster_colors: A list of color strings for plotting.
    - dic: A dictionary with tuple keys and list values.
    - dic2: A dictionary with tuple keys and list values.
    - adjust: A float that represents a scaling factor.
    - save: A boolean to save the plot image or not. Default is False.

    Returns:
    - dic2: A dictionary that has tuple keys and list values of the second set of data for the model.
    """
    last=0
    Up_max=[]
    for i in dic:
        f_index=i[0]
        if f_index is not last:
            for count,z in enumerate(Up_max):
                #note / 10 b/c of mote carlo
                plt.errorbar(count,z.mean(),yerr=z.std()/10,fmt="o",c=Cluster_colors[count])
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
            store_max.append((g[-1]))
        store_max=np.array(store_max)
        store_all=np.array(store_all)
        dic2[i].append((store_all/adjust))
        Up_max.append(store_max)
        last=f_index
    for count,z in enumerate(Up_max):
        #note the std / 10 is because of the net monte carlo
        plt.errorbar(count,z.mean(),yerr=z.std()/10,fmt="o",c=Cluster_colors[count])
    plt.title(f"Final Preformance for Base Cluster {i[0]}")
    plt.ylabel("R^2 Preformance on Test set")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Final_preformance{i}.png",dpi=400)
    plt.show()
    return dic2

def Transfer_graphs(dic2,resolution,epoch_conversions,Cluster_colors,byte,std_store,epochs,save=False):
    """
    A function to plot the transfer graphs.

    Args:
    - dic2: A dictionary with tuple keys and list values of the second set of data for the model.
    - resolution: An integer that represents the number of points to plot for the graphs.
    - epoch_conversions: A dictionary that has integer keys and string values.
    - Cluster_colors: A list of color strings for plotting.
    - byte: A boolean that indicates whether the data is in bytes.
    - std_store: A dictionary that has tuple keys and list values of the standard deviation matrix.
    - epochs: An integer that represents the number of epochs.
    - save: A boolean to save the plot image or not. Default is False.
    """
    dif_holder=[]
    std_diff=[]
    overfit_holder=[]
    residual_holder=[]
    std_res=[]
    integral_holder=[]
    std_int=[]
    error_holder=[]
    last=0
    difference=.05
    master={}
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
        sindex=np.where(sy<(difference+sy[-1]))
        #regular diffs
        diff=max(dic2[i][0])-min(dic2[i][0])
        std1=std_store[i][0]
        #overfit test
        overfit=(dic2[i][0][0]-dic2[i][0][-1]) > 0
        if f_index is not last:
            output1={}
            output2={}
            output3={}
            for count,z in enumerate(dif_holder):
                if std_diff[count] < 0:
                    std_diff[count]==0
                if overfit_holder[count]:
                    #plt.errorbar(count,-z,marker="x",c=Cluster_colors[count],yerr=std_diff[count])
                    t1="Over"
                    t1_1="Over"
                else:
                    #plt.errorbar(count,z,marker="o",c=Cluster_colors[count],yerr=std_diff[count])
                    t1=z
                    t1_1=std_diff[count]
                output1[count]=[t1,t1_1]
            plt.title(f"Average Training Deviation on the Worst Epoch for Base {i[0]-1}")
            plt.ylabel("R^2 deviation from best preforming model")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Training_deviation{i[0]-1}.png",dpi=400)
            plt.show()
            for count,g in enumerate(residual_holder):
                if g == "error":
                    #plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
                    t2="Over"
                    t2_1="Over"
                else:
                    conversion=epoch_conversions[count]
                    g=g*epoch_conversions[count]
                    plt.ylabel("Datapoints")
                    if GB:
                        conversion=conversion*byte*0.000001
                        g=g*byte*0.000001 #mega bytes
                        plt.ylabel("MegaBytes")
                    t2=g
                    t2_1=std_res[count]*conversion
                    if std_res[count] < 0:
                        std_res[count]==0
                    #plt.errorbar(count,g,c=Cluster_colors[count],yerr=std_res[count]*conversion,fmt="o")
                output2[count]=[t2,t2_1]
            plt.title(f"Average Information to reach {difference} Deviation Base {i[0]-1}")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Average_info{i[0]-1}.png",dpi=400)
            plt.show()
            for count,f in enumerate(integral_holder):
                if f == "error":
                    #plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
                    t3="Over"
                    t3_1="Over"
                else:
                    #plt.errorbar(count,f,c=Cluster_colors[count],yerr=std_int[count],fmt="o")
                    t3=f
                    t3_1=std_int[count]
                output3[count]=[t3,t3_1]
            plt.title(f"Learning Efficency to reach {difference} R^2 Deviation : Base {i[0]-1}")
            plt.ylabel("Net R^2 deviation squared")
            plt.xlabel("Transfer Cluster")
            if save:
                plt.savefig(f"Learning_eff{i[0]-1}.png",dpi=400)
            plt.show()
            dif_holder=[]
            std_diff=[]
            overfit_holder=[]
            residual_holder=[]
            std_res=[]
            integral_holder=[]
            std_int=[]
            master[(f_index-1,"set1")]=output1.copy()
            master[(f_index-1,"set2")]=output2.copy()
            master[(f_index-1,"set3")]=output3.copy()
        dif_holder.append(diff)
        std_diff.append(std1)
        overfit_holder.append(overfit)
        try:
            first=index[0][0]
            sfirst=sindex[0][0]
            integral=scipy.integrate.simps(y[:first], x=x[:first], dx=1, axis=-1, even='first')
            sintegral=scipy.integrate.simps(sy[:sfirst], x=sx[:sfirst], dx=1, axis=-1, even='first')
            integral_holder.append(integral)
            std_int.append(sintegral-integral)
            print("integral,std_int,+1_std,normal")
            print(integral,sintegral-integral,sx[sfirst],x[first])
            residual_holder.append(x[first])
            std_res.append(sx[sfirst]-x[first])
        except Exception as e: 
            print(e)
            print("error", i)
            integral_holder.append("error")
            std_int.append(0)
            residual_holder.append("error")
            std_res.append(0)
        last=f_index
    for count,z in enumerate(dif_holder):
        if std_res[count] < 0:
            std_res[count]==0
        if overfit_holder[count]:
            #plt.errorbar(count,-z,marker="x",c=Cluster_colors[count],yerr=std_diff[count])
            t1="Over"
            t1_1="Over"
        else:
            #plt.errorbar(count,z,marker="o",c=Cluster_colors[count],yerr=std_diff[count])
            t1=z
            t1_1=std_diff[count]
        output1[count]=[t1,t1_1]
    plt.title(f"Average Training Deviation on the Worst Epoch for Base {i[0]}")
    plt.ylabel("R^2 deviation from best preforming model")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Training_Deviation{i[0]}.png",dpi=400)
    plt.show()
    for count,g in enumerate(residual_holder):
        if g == "error":
            plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
            t2="Over"
            t2_1="Over"
        else:
            conversion=epoch_conversions[count]
            g=g*epoch_conversions[count]
            plt.ylabel("Datapoints")
            if GB:
                conversion=conversion*byte*0.000001
                g=g*byte*0.000001 #mega bytes
                plt.ylabel("MegaBytes")
            t2=g
            t2_1=std_res[count]*conversion
            if std_res[count] < 0:
                std_res[count]==0
            else:
                pass
                #plt.errorbar(count,g,c=Cluster_colors[count],yerr=std_res[count]*conversion,fmt="o")
        output2[count]=[t2,t2_1]  
    plt.title(f"Average Information to reach {difference} Deviation Base {i[0]}")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Average_info{i[0]}.png",dpi=400)
    plt.show()
    for count,f in enumerate(integral_holder):
        if f == "error":
            #plt.errorbar(count,0,c=Cluster_colors[count],yerr=0,fmt="x")
            t3="Over"
            t3_1="Over"
        else:
            #plt.errorbar(count,f,c=Cluster_colors[count],yerr=std_int[count],fmt="o")
            t3=f
            t3_1=std_int[count]
        output3[count]=[t3,t3_1]
    plt.title(f"Learning Efficency to reach {difference} R^2 Deviation : Base {i[0]}")
    plt.ylabel("Net R^2 deviation squared")
    plt.xlabel("Transfer Cluster")
    if save:
        plt.savefig(f"Learning_eff{i[0]}.png",dpi=400)
    plt.show()
    print(f_index,output3,[t3,t3_1])
    master[(f_index,"set1")]=output1
    master[(f_index,"set2")]=output2
    master[(f_index,"set3")]=output3
    return master

def make_distance_graph(dic,distances,Cluster_colors,name,save=False):
    last=0
    Up_max=[]
    master={}
    for i in dic:
        f_index=i[0]
        if f_index is not last:
            output={}
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            for count,z in enumerate(Up_max):
                temp=distances[f_index-1]
                #plt.errorbar(temp[count],z.mean(),yerr=z.std(),fmt="o",c=Cluster_colors[count],label=f"Cluster {count}")
                output[count]=[temp[count],z.mean(),z.std()]
            master[f_index-1]=output
            plt.title(f"Final Preformance for Base Cluster {i[0]-1}")
            plt.ylabel("R^2 Preformance on Test set")
            plt.xlabel(f"{name} Distance")
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            text = ax.text(-0.2,1.05, " ", transform=ax.transAxes)
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.13, 1))
            if save:
                plt.savefig(f"Distance_Graph_{i[0]-1}_{name}.png", bbox_extra_artists=(lgd,text), bbox_inches='tight')
            plt.show()
            Up_max=[]
        store_max=[]
        store_all=[]
        for count,g in enumerate(dic[i]):
            if count==0:
                store_all=np.array(g)
            else:
                store_all=store_all+np.array(g)
            store_max.append((g[-1]))
        store_max=np.array(store_max)
        store_all=np.array(store_all)
        #dic2[i].append((store_all/adjust))
        Up_max.append(store_max)
        last=f_index
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    output={}
    for count,z in enumerate(Up_max):
        temp=distances[len(distances)-1]
        plt.errorbar(temp[count],z.mean(),yerr=z.std(),fmt="o",c=Cluster_colors[count],label=f"Cluster {count}")
        output[count]=[temp[count],z.mean(),z.std()]
    print(f_index)
    master[f_index]=output
    ax.set_title(f"Final Preformance for Base Cluster {i[0]}")
    plt.ylabel("R^2 Preformance on Test set")
    plt.xlabel(f"{name} Distance")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    text = ax.text(-0.2,1.05, " ", transform=ax.transAxes)
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.13, 1))
    if save:
        plt.savefig(f"Distance_Graph_{i[0]}_{name}.png", bbox_extra_artists=(lgd,text), bbox_inches='tight')
    plt.show()
    return master

def distance_plots(df,y):
       ax=df.plot.scatter(x="Distance",y=y)
       z=getattr(df, y)
       for i, txt in enumerate(df.labels):
              ax.annotate(txt, (df.Distance.iat[i]+0.05, z.iat[i]))