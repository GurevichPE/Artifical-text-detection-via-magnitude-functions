# Artifical-text-detection-via-magnitude-functions
Final project for Selected Topics in Data Science course

Prepared by: Hassan Iftikhar, Pavel Gurevich

The aim of the project is to analyze the human-written and AI-generated texts using magnitude function. 

## Environment

To create the python environment run:

```
conda env create -f environment.yml
```

## Data preparation

We used a dataset with human and AI-generated texts from Kaggle. You can download it [here](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset).

To do a first try and test all the pipeline we took a subset of the whole dataset. To make it, you should run the `python prepare_subsample.py` code in the 'data_preprocessing' directory. Before running the code, specify `DATA_PATH` (path to your .csv file with dataset) and `SAVEDIR` (directory where you want to save your subsample). Also specify `SAMPLE_SIZE` -- number of elements for each class (if SAMPLE_SIZE=500, the final subsample will be size of 1000: 500 for human-written and 500 for AI-generated texts).

Next, to prepare text embeddings you should run the `python get_embeddings.py` code in the 'data_preprocessing' directory. Here also don't forget to specify global variables before running. This code will prepare embeddings of your texts using BERT model and save them and corresponging class labels as pytorch tensors (.pt files).

## Magnitude function implementation

Script `magnitude_function.py` in the 'magnitude' folder implements calculation of magnitude and magnitude function. The example of its work is in the `experiments.ipynb` notebook in the same directory. 
