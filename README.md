# Artifical-text-detection-via-magnitude-functions
Final project for Selected Topics in Data Science course

Prepared by: Hassan Iftikhar, Pavel Gurevich

The aim of the project is to analyze the human-written and AI-generated texts using magnitude function. 

## Environment

To create the python environment run:

```bash
conda env create -f environment.yml
```

## Data preparation

We used a dataset with human and AI-generated texts from Kaggle. You can download it [here](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset).

To do a first try and test all the pipeline (and also because of memory and resources limitations) we took a subset of the whole dataset. To make it, you should run the `python prepare_subsample.py` code in the 'data_preprocessing' directory. Before running the code, specify `DATA_PATH` (path to your .csv file with dataset) and `SAVEDIR` (directory where you want to save your subsample). Also specify `SAMPLE_SIZE` -- number of elements for each class (if SAMPLE_SIZE=500, the final subsample will be size of 1000: 500 for human-written and 500 for AI-generated texts).

Next, to prepare text embeddings you should run the `python get_embeddings.py` code in the 'data_preprocessing' directory. Here also don't forget to specify global variables before running. This code will prepare embeddings of your texts using BERT model and save them and corresponging class labels as pytorch tensors (.pt files).

To calculate magnitude function(s) use `get_magnitude_function.py`. Before running, specify global variables, which are paths to embeddings and save directory, parameters of `t` grid, magnitude metric (see [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) documentation for the whole list of available metrics), `N_JOBS` for multiprocessing (it's really reducing the computational time) and list of `MAX_TOKEN_LENGTHS`. If you calculated embeddings for only one max token length, set just list with one value (like `MAX_TOKEN_LENGTHS=[288]`).  

## Magnitude function implementation

Script `magnitude_function.py` in the 'magnitude' folder implements calculation of magnitude and magnitude function. 

## Logistic Regression training and evaluation

To run the Logistic regression training and evaluation, execute the `train.py` script in the 'magnitude_classifier' directory. 

Specify the path to save directory, directories with magnitude function files, embeddings, labels and dataset. Specify the working mode: `magnitude` to run LogReg on magnitude functions only; `embeddings` to run LogReg on embeddings only, and `both` to run LogReg on the concatenated embeddings and magnitude functions. `MAX_TOKEN_LENGTHS` is the same as in the `get_magnitude_function.py`. Finally specify `N_STEPS` for proper downloading of magnitude function files.

This code runs stratified 6-fold CV Logistic Regression and saves the .pkl file of the dictionary with the following keys: 

+ 'aucs' : list of ROC AUC values corresponding for each value of `MAX_TOKEN_LENGTHS`
+ 'mistakes' : list of dictionaries. One dictionary contains the LLM model names as keys and number of misclassification (how much the LogReg classified this model's text as human-made one) as values.
+ 'len_grid' :  `MAX_TOKEN_LENGTHS`. Just for logging.


## For the most curious

Also you can check the following raw (not nice, not polished) jupyter notebooks to see some additional experiments or plotting examples:

+ `experiments.ipynb` (in the project root, yep). Provides experiments with Text length, Token window size, magnitude function augmentation with some additional features, and checking the work of calculating magnitude with Conjugate Gradient algorithm (find the `gradient_z_inverse` function)
+ `magnitude/experiments.ipynb`: Plotting for magnitude function parametrization (experiments with parameter `t`).
+ `magnitude_classifier/experiments.ipynb`: plotting of LogReg results. Like misclassifications, ROC AUCs. 
