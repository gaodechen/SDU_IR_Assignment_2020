## SDU Information Retrieval Assignment

**SDU IR Assignment Spring, 2020**

**Documents Re-ranking**

## Problem

Documents re-ranking using raw text as dataset.

Recommend another re-ranking implementation based on **BERT** from [zhangt2333/SDUIR2020-Experiment](https://github.com/zhangt2333/SDUIR2020-Experiment).

## Method

### Feature Engineering

* Preprocess raw text file and convert into dataframes
* Get features of BM25, TF-IDF, etc.
* Construct dataset with given ratio of negative samples.

### Training & Blending

#### Base models

* LightGBM
* XGBoost
* CatBoost

#### Tuning

* Grid search
* Bayesian Optimization

Details in IR_Assignment.pdf.