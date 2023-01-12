Data_get
==============================

Data download file for the NYISO database.  Supports gen and zonal data currently.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. explaing main ways to download data
    │   └── functions      <- Has functions for 2 subclasses 
    │   │      └── data    <- Has functions related to formatting downloaded data
    │   │      └── driver  <- Has functions related to downloading data
    │   └── local          <- functions that refer to local pathing before upload and formatting
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
Organized from cookie cutter, see no need to remove the general form of above figure.  not all functionalities are currently being used

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