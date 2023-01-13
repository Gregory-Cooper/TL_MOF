import pandas as pd
import os
from sklearn import preprocessing
# Note that these carried from previous paper 

def data_loader(base_path, file_name):
    """
    loads data from specifc mof dataset 

    Inputs 
    ------
    Base_path (str) - locaction of directory
    file_name (str) - name of the file

    Output 
    ------
    data (df) - df of MOF data
    """
    directory = str(os.path.join(base_path, file_name))
    data = pd.read_csv(directory, sep=',', header=None)
    data.columns = ['MOF ID', 'void fraction', 'Vol. S.A.', 'Grav. S.A.', 'Pore diameter Limiting',
                    'Pore diameter Largest',
                    'H2@100 bar/77K (g/L)', 'H2@100 bar/77K (mol/kg)', 'H2@100 bar/77K (wt%)',
                    'H2@100 bar/130K (g/L)', 'H2@100 bar/130K (mol/kg)', 'H2@100 bar/130K (wt%)',
                    'H2@100 bar/200K (g/L)', 'H2@100 bar/200K (mol/kg)', 'H2@100 bar/200K (wt%)',
                    'H2@100 bar/243K (g/L)', 'H2@100 bar/243K (mol/kg)', 'H2@100 bar/243K (wt%)',
                    'Qst@6 bar/77K (H2)', 'Qst@6 bar/130K (H2)', 'Qst@6 bar/200K (H2)', 'Qst@6 bar/243K (H2)',
                    'CH4@100 bar/298 K (v/v)', 'CH4@100 bar/298 K (mg/g)',
                    'CH4@65 bar/298 K (v/v)', 'CH4@65 bar/298 K (mg/g)',
                    'Qst@6 bar/298K (CH4)',
                    '1 bar Xe mol/kg', '1 bar Kr mol/kg', '1 bar Selectivity',
                    '5 bar Xe mol/kg', '5 bar Kr mol/kg', '5 bar Selectivity',
                    'topology',
                    'First nodular symmetry code', 'First nodular character', 'First nodular ID',
                    'Second nodular symmetry code', 'Second nodular character', 'Second nodular ID',
                    'Connecting building block ID', 'Nonsense']
    # Remove the last nonsense column
    data = data.drop(['Nonsense'], axis=1)
    # Remove the row which contains null value
    data = data.dropna(axis=0)

    return data


def all_filter(data, filter_columns):
    """
    filters features from df

    Inputs 
    ------
    data (df) - df of MOF data
    filter (array of str /strs) - features that you want filtered

    Output 
    ------
    all_property (df) - df of MOF data without specific features
    """    
    all_property = data.drop(filter_columns, axis=1)

    return all_property


def get_descriptors(data, descriptor_columns):
    """
    Gets Features of interest from df

    Inputs 
    ------
    data (df) - df of MOF data
    descriptor_columns (array of str /strs) - features that you want

    Output 
    ------
    descriptors (df) - copy of descriptor columns
    """   
    descriptors = data[descriptor_columns].copy()

    return descriptors


def one_filter(data, one_filter_columns):
    """
    filters features from df

    Inputs 
    ------
    data (df) - df of MOF data
    one_filter_columns (array of str /strs) - features that you want 

    Output 
    ------
    all_property (df) - feature of interest
    """    
    one_property = data[one_filter_columns].copy()

    return one_property


def data_scaler(data):
    """
    standard scales data 

    Inputs 
    ------
    data (df) - df of MOF data

    Output 
    ------
    scaled data
    """    
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(data)
