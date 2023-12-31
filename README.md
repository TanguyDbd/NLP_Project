<div align="center">

# ***Natural Language Processing Project October/November 2023 EPF Montpellier Data Engineering Major***
<h1 align="center">
<br>RESTAURANT REVIEW CLASSIFICATION</h1>
</div>

##  1. Table of Contents
- [***Natural Language Processing Project October/November 2023 EPF Montpellier Data Engineering Major***](#natural-language-processing-project-octobernovember-2023-epf-montpellier-data-engineering-major)
  - [1. Table of Contents](#1-table-of-contents)
  - [2. Introduction](#2-introduction)
  - [3. Repository Structure](#3-repository-structure)
  - [4. Installation](#4-installation)
  - [5. Modules](#5-modules)
    - [1. Exploratory Data Analysis Notebook](#1-exploratory-data-analysis-notebook)
    - [2. Preprocessing python file](#2-preprocessing-python-file)
    - [3. Baseline Model + Improvement Notebook](#3-baseline-model--improvement-notebook)
    - [4. Deep Learning Notebook](#4-deep-learning-notebook)
  - [6. Performance Table](#6-performance-table)
  - [7. Conclusion](#7-conclusion)
  - [8. Acknowledgments](#8-acknowledgments)

---

##  2. Introduction
This project is designed to go through various aspects of a professional Natural Language Processing (NLP) project. It aims to classify restaurant reviews submitted by different customers for various restaurants. The dataset, sourced from Kaggle that you can retreive [here](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews), consists of 10,000 reviews, each associated with a rating ranging from 1 to 5. The objective is to construct an NLP model capable of predicting future ratings based on given reviews.

To achieve this goal, the project aims to develop the most effective predictive model, emphasizing performance across various steps outlined below.

This project takes part of my fith year at EPF Engineering School. It summarized what I learned during my NLP class during October and November 2023.

---

##  3. Repository Structure

```
└── NLP_Project/
    ├── Restaurant reviews.csv
    ├── Exploratory_Data_Analysis.ipynb
    ├── Baseline_Model_and_Improvement.ipynb
    ├── Deep_Learning_Model.ipynb
    ├── requirements.txt
    ├── mlModel.py
    └── preprocessing_function.py

```

---

##  4. Installation

1. Clone the NLP_Project repository:
```sh
git clone https://github.com/TanguyDbd/NLP_Project
```

2. Change to the project directory:
```sh
cd NLP_Project
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

---

##  5. Modules
This project is made of 3 different notebooks, a python file and a csv file as dataset.

### 1. [Exploratory Data Analysis Notebook](Exploratory_Data_Analysis.ipynb)
This first notebook `Exploratory_Data_Analysis.ipynb` is about a first exploration of the dataset where I tried to plot relevant inofrmation about it and the different manipulation that had to be made to allow further analysis.

### 2. [Preprocessing python file](preprocessing_function.py)
This python file `preprocessing_function.py` containes a preporcessing script that can be change at any moment. So far it tokenizes the data, remove the punctuation and convert to lowercase, remove stopwords and lemmatizes the data. It also contains another function that removed the 10 most common words in my text data.

### 3. [Baseline Model + Improvement Notebook](Baseline_Model_and_Improvement.ipynb)
In this second notebook `Baseline_Model_and_Improvement.ipynb`, we import our preprocessing pipeline, apply it to our dataset and train a machine learning model without any particular parameter tuning or feature engineering. The goal here is simply to obtain a baseline model which we'll use as reference for future experiments.

Then, we try to optimize this model by doing different GridSearch on first the Vectorizer/Model Architecture combination, before doing some on the hyperparameters of the best model we have so far. The goal here is to improve the accuracy of the model.

### 4. [Deep Learning Notebook](Deep_Learning_Model.ipynb)
After trying to improve our baseline model, we want to add deep learning skills to our model in `Deep_Learning_Model.ipynb`. In this notebook, we tried several Deep Learning models architecture to find if we get a better accuracy

---

##  6. Performance Table

| Model                        | Accuracy (%) | Training size | Number of epochs | Training time per epoch (s) |
|------------------------------|--------------|---------------|------------------|-----------------------------|
| [Basic Model](#-performance-table)          | 64           | 9817          | /              | /                         |
| [Simple Neural Network](#-performance-table)        | 55           | 9817          | 10              | 37                          |
| [RNN](#-performance-table)                          | 51           | 9817          | 10              | 110                         |
| [LSTM](#-performance-table)                         | 58           | 9817          | 10              | 220                          |




---

##  7. Conclusion
First, doing this project was very interesting. Taking the time to do a full NLP project and see the whole process is very interesting no matter how good my results are. This helped me to apply what we have learned during this class and triggered my attention on this very interresting subject.

Regarding them, they looks quite insufisant but the main hypothesis I think about is because of the complexity of my dataset which is unbalanced : the different rates don't have the same number of reviex at all and the distribution is quite messy. The next steps will be to try to oversample my model in order to balance my data.

Also, with more epochs, I am convince that my deep learning models would perform better even with the same dataset but the ones I worked with already took a long time to execute. Regarding the fact that no matter what improvements I brought the model's performances didn't improve, I might have done mistakes that could explain those results.

Finally, even if my project isn't as complete as I would want to, it helped me understand better the Natural Language Processing environement that I hope I will have the opportunity to work with in a way or another.

---

##  8. Acknowledgments
- [Ryan Pegoud](https://github.com/RPegoud) for his clear lessons, which enabled me to discover Natural Language Processing in the best possible way.
- Kaggle for the dataset availability
- Thalita Drumond and Robert Rapadamnaba for their different courses on the theory behind Machine Learning and Deep Learning models
- ChatGPT for the help to implement correctly my models specially the Deep Learning ones because of their complexity