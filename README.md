Metal-Organic Framework Clustering through the Lens of Transfer Learning
==============================

Abstract
------------
Metal-organic frameworks (MOFs) are promising materials currently being studied for various applications using machine learning (ML) techniques that enable their design and understanding of structure-property relationships. This work adds insight into the MOF landscape and the use of ML within it. We cluster the MOFs using two different approaches. For the first set of clusters, we use principal component analysis (PCA) on the textural properties (void fraction, surface area, pore size, etc.) and cluster the resulting components using k-means. We also separately cluster the MOF space using agglomerative clustering based on their topologies. The feature data from each of the clusters were then fed into separate neural networks (NNs) for direct learning on an adsorption task (methane or hydrogen). The resulting NNs were used in transfer learning (TL) where only the last layer was retrained on data from different clusters but the same adsorption task. The results show significant differences in TL performance (R2, standard deviation, etc.) based on which cluster is chosen for direct learning. We find TL performance depends on the Euclidean distance in the principal component space between the clusters involved in the direct and transfer learning. Similar results were found when TL was performed simultaneously across both types of clusters and adsorption tasks. Interestingly, we find methane adsorption outperforms hydrogen adsorption as the source task. Our study opens the door to understanding how MOFs relate to each other and establishes a method to identify privileged structures capable of transferring information to other parts of the MOF landscape.

Data set
------------
The data set used in this work was originally generated in an older report and has also been used previously in a transfer learning study. Briefly, over 13,000 MOF structures were computationally generated using the topologically based crystal constructor (ToBaCCo). They represent a diverse set of structures from a topology perspective. The textural properties of the structures were also determined using a variety of tools. Lastly, GCMC simulations were performed to determine methane and hydrogen adsorption at various conditions. In this work, we use the topology of the structures and the textural properties for clustering, the textural properties for training neural networks in direct and transfer learning, and the adsorption of methane and hydrogen as the tasks to be learned. The data set can be obtained from the relevant publications and github repositories associated with them. 



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
    ├── Cluster_analysis.ipynb <- file used to analize the data set and make most general figures
    ├── mover.ipynb <- packages any output pngs into a file   
    ├── process_json_GC.ipynb <- processes json output of any of the GC cluster outputs (created by files in generic files or data_json generic)
    ├── process_json_top.ipynb <- same as above but with topology
    └── transfer learning.py        <- functions used to create general nn classes used in processing
    

--------


Instructions
------------
This project is suppose to get NYISO data from the site and format it for later use in projects.  As such there are two main steps.  The first is to actually use a scapper to get the data of the site, the second is to format the data for reupload. 

   1. Download
   
            The first step is to use the webdriver_explained file to work and download the data. 
            All the needed function are found in the data submodule so if you want a deeper dive check there. See figure one for
            imformation on how to set up the files ideally for download.  This will make the
            process much easier when looking at os and other files later.  You can change this pathing and layout, but currently is no 
            supported by the code. 
        
   2. Rename locally 
   
            The second step is to take the data and rename it into similar formatting so that y
            ou can loop through csv later.  Edge will do this to 100 so you can theoretically, but that requires some changing
            of functions. This is done in the local folder in os_rename for my method.
        
   3. Reformat for upload
   
            The last step is to finish the formating of the actual underlying data and reupload it as
            a set of csv.  The best way to do this is to loop through all of them thus they need to be generalize as in step 2.
            The actual process is described in format_explained and the functions are found in data folder 
            for the submodule.  It will output in form of figure 2 recommended.  If using large data set and 
            memory is a issue see large_data_explained for making only the maindf portion
            Any other is not supported outright by the code
        
   4. Upload (optional)
   
            Upload data to wherever you see fit

Figure 1        
--------

    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
            └── type (rt-gen)    <- The type of data (ex real time generator abr. as rt_gen)
                └── 2010            <- The first year of data you want (2010 choosen for this data thus far)
                └── ....            <- middle years between
                └── 20xx            <- The last year of data you want (2020 choosen for this data thus far for reference)
 
Figure 2        
--------

    ├── data
        └── interim        <- Intermediate data that has been transformed.
            ├── maindf      <- The non main df set (1 year per gen per csv)
            │   ├── 2010            <- The first year of data you want (2010 choosen for this data thus far)
            │   ├── ....            <- middle years between
            │   └── 20xx            <- The last year of data you want (2020 choosen for this data thus far for reference)
            │       ├── gen1.csv            <- The first generator of data (will be called whatever gen name is in df and subsequent)
            │       ├── ....                <- intermediate gen
            │       └── genx.csv            <- The last gen you want 
            │
            │
            └── masterframedf    <- masterframe formatted set (10 year in 1 csv per gen)
                ├── gen1.csv            <- The first generator of data (will be called whatever gen name is in df and subsequent)
                ├── ....                <- intermediate gen
                └── genx.csv            <- The last gen you want 

cookie cutter reference --- href="https://drivendata.github.io/cookiecutter-data-science/">