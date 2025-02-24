Metal-Organic Framework Clustering through the Lens of Transfer Learning
==============================

Abstract
------------
Metal–organic frameworks (MOFs) are promising materials with various applications, and machine learning (ML) techniques can enable their design and understanding of structure–property relationships. In this paper, we use machine learning (ML) to cluster the MOFs using two different approaches. For the first set of clusters, we decompose the data using the textural properties and cluster the resulting components. We separately cluster the MOF space with respect to their topology. The feature data from each of the clusters were then fed into separate neural networks (NNs) for direct learning on an adsorption task (methane or hydrogen). The resulting NNs were then used in transfer learning (TL) where only the last NN layer was retrained. The results show significant differences in TL performance based on which cluster is chosen for direct learning. We find TL performance depends on the Euclidean distance in the decomposed feature space between the clusters involved in the direct and TL. Similar results were found when TL was performed simultaneously across both types of clusters and adsorption tasks. We note that methane adsorption was a better source task than hydrogen adsorption. Overall, the approach was able to identify MOFs with the most transferable information, leading to valuable insights and a more comprehensive understanding of the MOF landscape. This highlights the method's potential to generate a deeper understanding of complex systems and provides an opportunity for its application in alternative datasets.

Data set
------------
The data set used in this work was originally generated in an older report and has also been used previously in a transfer learning study. Briefly, over 13,000 MOF structures were computationally generated using the topologically based crystal constructor (ToBaCCo). They represent a diverse set of structures from a topology perspective. The textural properties of the structures were also determined using a variety of tools. Lastly, GCMC simulations were performed to determine methane and hydrogen adsorption at various conditions. In this work, we use the topology of the structures and the textural properties for clustering, the textural properties for training neural networks in direct and transfer learning, and the adsorption of methane and hydrogen as the tasks to be learned. The data set can be obtained in the data file lists or in that noted in acknowledgements.



Project Organization
------------

** Note - removed from paper = (rfp)**

    ├── README.md          <- The top-level README for developers using this project.
    ├── Batch_Runs          <- Data regarding optimizing the batch size
    │   ├── (*).py      <- Any File used to run on cluster for specific batch
    │   │
    │   ├── Batch_base_code.ipynb     <- Base code used to generate .py files (mostly for reference) 
    │   │
    │   ├── Sherpa_batch_optimize.ipynb     <- uses code above now extended to use sherpa optimization package  
    │   │ 
    │   └── Data_Batch            <- Raw data and analysis files
    ├── Data          <- Data (see dataset)
    │   └── data.csv            <- Tabular MOF Data 
    ├── Data_for_comparision          <- Data (see dataset)
    │   ├── (*).pickle      <- Specifc data for each batch process
    │   │
    │   └── Compare_smaples.ipynb            <- Used to analyze the data on each model that one run created and compare with more granularity.
    ├── Data_Json          <- Processed Json output of each trial
    │   ├── Generic      <- Data for Generic cluster trials (Transfer refers to task transfer)
    │   │
    │   ├── Replicability_test     <- Data using same seed 2 times to ensure replicability
    │   │
    │   ├── SubCluster_Depreciated     <- Data for sub-topology transfer (rfp)
    │   │
    │   └── Topology            <- Data for Topology cluster trials (Transfer refers to task transfer)
    ├── Depreciated_Work          <- Depreciated methods, files for quick anaylsis, or old prototypes
    │   ├── 4_clusters_depreciated      <- Using 4 clusters for topology instead of 6 
    │   │
    │   ├── MISC anaylsis     <- Files associated with analysis of data that were phased out
    │   │
    │   ├── Prototypes     <- Files that were used to build from.  Includes optimizations, sub-topology, etc
    │   │
    │   └── process_json_subcluster.ipynb            <- File for processing subclusters of topologies (rfp)
    ├── Functions          <- Functions directly used in analysis or run files for paper (organized by name)
    │   ├── dataset_loader.py 
    │   ├── engine.py
    │   ├── helper.py    
    │   └── Statistics_helper.py
    ├── Generic Files          <- Files for generic cluster data generation
    │   └── (*).ipynb            <- file to generate specific trial data associated with respect to the file name
    ├── One_Run          <- Files associated with producing one trial of the data (Not Monte Carlo)
    │   └── (*).ipynb            <- file to generate specific trial data associated with respect to the file name
    ├── Optimize_absorbates          <- Files associated with finding optimal parameters for each task
    │   ├── Optimize_models      <- folder of pre optimized nn models for each task
    │   │
    │   ├── (*) OPT.ipynb      <- Files used to generate optimized data and models
    │   │
    │   ├── (*) OPT.json      <- optimized model data
    │   │    
    │   └── hyper(*).ipynb            <- files assocaited with analyzing each optimized model
    ├── Topology_GC_relationships          <- plots of Generic cluster to topology relationships
    ├── Topology Files          <- Files for Topology cluster data generation
    │   └── (*).ipynb            <- file to generate specific trial data associated with respect to the file name
    ├── Topology_histograms          <- Figures for each topology histograms
    ├── topology_outline          <- specific topology scatter in pca figures
    │  
    │ (*** Below are main files used to process data ***)
    ├── Cluster_analysis.ipynb      <- file used to analize the data set and make most general figures
    ├── mover.ipynb <- packages any output pngs into a file   
    ├── process_json_GC.ipynb       <- processes json output of any of the GC cluster outputs (created by files in generic files or data_json generic)
    ├── process_json_top.ipynb      <- same as above but with topology
    └── transfer learning.py        <- functions used to create general nn classes used in processing
    

--------


Instructions for running a trial
------------
The general way to run a run of the data / code below  

   1. Aquire the data

    1. Get the Data from json

            Simply pull the corresponding json from the Data_Json Folder

    2. Create the Data

        1. Pull the ipynb file from either topology or generic files and place into the net folder directory

        2. Make any changes that you want to the file to complete the run (change file configuration in ipynb)

                (Note that this will create display the configurations you need to process files later)

        3. Run the file through and generate the json file needed
    
        
   2. Used the correct process file

    1. Ensure that the topology files are used in process topology or vice versa for process generic (agreement with data)

    2. Enter the relevent configuration information in the first part of the code (this is commented out)

    3. It will output the needed figure set for that file
        
   3. Understanding the output
    
        The file will have the directory seen in Figure 1 below
        
Instructions for generating general figures
------------
The general way to run analysis of the dataset without processing

   1. Simply Run the cluster analysis.ipynb file in the directory 

        Note - this file should be adjusted for what one figures you want and is set to save the figures in the directory

   2. Use mover to package the output figures if desired

Figure 1        
--------

    ├── (Name of json File used in process)
    │   ├── META       <- meta figures associated with the average base clusters
    │   │
    │   ├── PC1       <- Figures where metrics are compared with respect to PC1
    │   │
    │   ├── PC2       <- Figures where metrics are compared with respect to PC2
    │   │
    │   ├── regular      <- Figures where metrics are compared with respect to net PC distance
    │   │
    │   └── Tables            <- Tabular summary of the data using tables with respect to net PC distance


Acknowledgements      
--------
Data was forked off the following database as a starting point: 

https://github.com/RUIMINMA1996/TL_MOF'

Which was used in conjunction with this paper :

[Transfer Learning Study of Gas Adsorption in Metal-Organic Frameworks](https://pubs.acs.org/doi/10.1021/acsami.0c06858?ref=pdf)
Ruimin Ma, Yamil J. Colón, Tengfei Luo

Creator Contacts
--------

Colón Group - [Website](https://www.computationalnano.org/)

Gregory Cooper -  [My Github](https://github.com/Gregory-Cooper) - coopergregorym@gmail.com

Yamil Colón -  ycolon@nd.edu

## Note
Code and data for academic purpose only.
 
