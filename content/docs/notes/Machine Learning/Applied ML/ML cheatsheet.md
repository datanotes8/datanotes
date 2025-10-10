# ML General Cheatsheet

## Cross-validation
Cross-validation is a technique used to assess how the results of a statistical analysis will generalize to an independent dataset. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice.

## Overfitting vs Underfitting
Overfitting: when model fits training data too well including outliers and pattterns. This results in poor generalization for new data. 
- High training data accuracy but low test data accuracy.

Methods to reduce overfitting:
1. Reduce the number of features
2. Add regularization terms (L1 or L2 regularization) 

Underfitting: model lacks sufficient generalization and did not capture underlying patterns in the data.
- Low training data accuracy and low test data accuracy.

Methods to reduce underfitting:
1. Increase training data or parameters
2. Increase model features
3. Reduce regularization

## Bias and variance trade-off
- Bias: error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
- Variance: error due to excessive sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).  


## Regularization
Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. Common types of regularization include:
- L1 Regularization (Lasso): adds the absolute value of the magnitude of coefficients as a penalty term to the loss function. It can lead to sparse models where some coefficients are exactly zero.
- L2 Regularization (Ridge): adds the squared magnitude of coefficients as a penalty term to the loss function. It tends to shrink coefficients but does not set them to zero.  

