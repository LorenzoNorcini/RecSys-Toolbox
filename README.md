# RecSys Toolbox

## Requirements
| Package              | Version        |
| ---------------------|:--------------:|  
| **scikit-learn**     |   >= 0.19.1    |   
| **numpy**            |   >= v1.14     |   
| **scipy**            |   >= 1.0.0     |   
| **pandas**           |   >= 0.22.0    |   
| **tqdm**             |   >= 4.19.5    |  

## Usage

Single Method Recommender

```python
from recommender import *

# urm is the user rating matrix in the format users x items
bcbf = BasicContentBasedFiltering(user_rating_matrix=urm)

# icm is the item content matrix in the format item x features
bcbf.fit(item_content_matrix=icm, k_nearest_neighbours=knn)

# target is the vector of interactions for a user
bcbf.predict(target)
```

CF + CBF + SLIM BPR + ALS + SVD Hybrid Recommender

```python
from recommender import *

bcbf = MixedHybrid(user_rating_matrix=urm, slim_lr=lr, slim_epochs=e, slim_pir=pir, 
                  slim_nir=nir, slim_knn=knn_slim, cbf_weight=al,cf_weight=be,
                  als_weight=te, svd_weight=de, slim_weight=ga)

# als and svd are precomputed similarity matrices of the shape
# items x items calculated from factorization
bcbf.fit(item_content_matrix=icm, k_nearest_neighbours=knn, computed_als=als, computed_svd=svd)

bcbf.predict(target)

```

## Challenge

These algorithms were implemented during the ["Recommender System 2017 Challenge Polimi"](https://www.kaggle.com/c/recommender-system-2017-challenge-polimi) for the Recommender System Course at Politecnico di Milano.

The task of the challenge was to predict what songs would be added to the users' playlists in the future.

The classes and methods provided in the "challenge.py" and "preprocessing.py" refer to the preprocessing of the dataset and the predictions for the submission file.

Given the characteristics of the challenge the algorithms implemented in the "recommender.py" file are intended for implicit feedback.

More examples that use the challenge dataset can be found in the "examples.py" files.

The data is available [here](https://www.kaggle.com/c/recommender-system-2017-challenge-polimi/data).

## Results

The best performance has been obtained with an Hybrid Model that combined the cosine similarities between items obtained from 
* Basic Content Based Filtering
* Basic Collaborative Filtering
* Alternating Least Squares Matrix Factorization
* Singular Value Decomposition

The ratings obtained using this similarity are combined with the ratings obtained using
* SLIM Bayesian Personalized Ranking

## Score

The performance is evaluated using Mean Average Precision at 5.

The best model scored MAP@5 = 0.10280

## Model hyperparameters

| Parameter            | Value          |
| ---------------------|:--------------:|
| SVD Latent Factors   |   5000         |
| SVD Nearest Neighbours              |   600          |
| ALS Latent Factors   |   5000         |
| ALS Nearest Neighbours            |   1000         |
| ALS Regularization   |   0.1          |
| ALS Iterations       |   15           |
| Combined Similarity Nearest Neighbours | 300|
| SLIM BPR Learning Rate | 0.01   |
| SLIM BPR Epochs | 1|
| SLIM BPR Nearest Neighbours | 600 |
| SLIM BPR Positive Item Regularization | 1 |
| SLIM BPR Negative Item Regularization | 1 |
| CBF Similarity Weight   | 0.8 |
| CF Similarity Weight    | 0.2 |
| ALS Similarity Weight   | 0.07|
| SVD Similarity Weight   | 0.7 |
| SLIM BPR Ratings Weight | 0.8 |
