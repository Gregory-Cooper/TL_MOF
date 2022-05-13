from Functions.dataset_loader import data_loader, get_descriptors, one_filter, data_scaler
import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from .Statistics_helper import stratified_cluster_sample
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


