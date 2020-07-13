## SDU Information Retrieval Assignment

**SDU IR Assignment Spring, 2020**

**Documents Re-ranking**

**Recommend another re-ranking implementation** based on BERT from [zhangt2333/SDUIR2020-Experiment](https://github.com/zhangt2333/SDUIR2020-Experiment).

## Problem

Re-ranking probelm given queries and documents in raw text format.

Denote query as $q$ and documents set as $D = \{d_1, d_2, ... , d_n\}$. our model should generate a re-orderer $D'$ by consideration of relavance between given query and each $d_i$.

### Data Format

#### Documents in .json

```
{
    d1_id: d1_text,
    d2_id: d2_text,
    ...
}
```

#### Training Data in .json

```
{
    queries: {
        q1_id: q1_text,
        q2_id: q2_text,
        ...
    },
    labels: {
        q1_id: [dx_id,...],
        q2_id2: [dx_id,...],
        ...
    }
}
```

#### Training Data in .csv

Moreover, we implement a [SVM-rank](http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)-like format after processing raw text input in .json format described above and computation of features, similarities.

```
target_i doc_id_i query_id_i feature_i1 feature_i2 ... feature_in
```

Each $feature_i$ computing with $q_i$ and $d_i$ combined would be the actual input of the model. Here the target values implicitly denote a pairwise preference of documents for $q_i$, **not** absolute values as learning targets.

## Method

Details in **IR_Assignment.pdf**.

We conducted a two-stage scheme including recalling & re-ranking to solve the issue. Specifically, **BM25** is selected in recalling stage and **blending on boosting models** is implemented in re-ranking stage.

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

* Grid Search Cross-validation
* Bayesian Optimization

## P.S.

The feasibility of this repo has been verified by invoking classical ways of features engineering and boosting trees, but practical usage still needs promotion with more dataset and better designing & coding.

We only got 500,000 documents and 20,000 queries as training data, which may lead to a less representative test result. So we don't recommend methods with specialized optimization onto this dataset.
