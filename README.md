# ***Natural Language Processing Project of Tanguy Dabadie***


# ***This project isn't finsh yet !***


## Introduction
This Project is made to cover the different aspect of a professionnal NLP project. It made to classify reviews of restaurant made by different customers on differnet restaurants. The dataset retreived on Kaggle contains 10000 reviews with a giving rate going from 1 to 5.

This project takes part of my fith year at EPF Engineering School. It summarized what I learned during my NLP class durign October 2023.

## Repository Composition
This project is made of 2 different notebooks, a python file and a csv file as dataset.

### 1. Exploratory Data Analysis Notebook
This first notebook named "1st_notebook" is about a first exploration of the dataset where I tried to plot relevant inofrmation about it and the different manipulation that had to be made to allow further analysis.

### 2. Preprocessing python file
This python file called preprocessing_function.py containes a preporcessing script that cand be chnage at any moment. So far it tokenizes the data, remove the punctuation and convert to lowercase, remove stopwords and lemmatizes the data. It also contains another function used to produce reports on machine learning models that we use in the second notebook.

### 3. First Model Notebook
In this second notebook named "2nd_notebook", we import our preprocessing pipeline, apply it to our dataset and train a machine learning model without any particular parameter tuning or feature engineering.

The goal here is simply to obtain a baseline model which we'll use as reference for future experiments. This where I am so far : the next steps will be optimising and improving it before applying deep learning skills to the problem that we will fin in the third and last notebook.

### 4. Deep Learning Notebook
To be done...

## Acknowledgments
- Ryan Pegoud for his clear lessons, which enabled me to discover Natural Language Processing in the best possible way.
- Kaggle for the dataset availibility