---
title: "ML Cheatsheet"
draft: false
---
# Core ML Concepts
---
## Cross-validation
Cross-validation is a technique used to assess how a model will generalize to an independent dataset. Another word for this is backtest using historical or actual data.
Common types:
- **K-Fold:** Split data into *k* parts, train on *k–1*, validate on 1.
- **Leave-One-Out (LOO):** Extreme form of K-fold with k = n.
- **Stratified K-Fold:** Keeps class distribution constant in each fold.
---
## Overfitting vs Underfitting
Overfitting: when model fits training data too well including outliers and pattterns. This results in poor generalization for new data. 
- High training data accuracy but low test data accuracy.
- Reduce by add regularization, features, or regularization (L1, L2)

Underfitting: model lacks sufficient generalization and did not capture underlying patterns in the data.
- Low training data accuracy and low test data accuracy.
- Reduce by add complexity or features, add data, reduce regularization
---
## Bias and variance trade-off
Error decomposition:
$$
\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

- Bias: error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
- Variance: error due to excessive sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting). 
---
## Regularization
Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. Common types of regularization include:
- L1 Regularization (Lasso): adds the absolute value of the magnitude of coefficients as a penalty term to the loss function. It can lead to sparse models where some coefficients are exactly zero.
- L2 Regularization (Ridge): adds the squared magnitude of coefficients as a penalty term to the loss function. It tends to shrink coefficients but does not set them to zero.  
---
## Scaling and normalization
- Scaling: transforming features to a specific range, such as [0, 1] or [-1, 1]. Common methods include Min-Max Scaling and Standardization (Z-score normalization).
- Normalization: transforming features to have a mean of 0 and a standard deviation of 1. This is often done using StandardScaler in libraries like scikit-learn.
---
## Metrics
- Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²)
    - MAE = $$\frac{1}{n} \sum | \text{Actual} - \text{Predicted} |$$
    - MSE = $$\frac{1}{n} \sum ( \text{Actual} - \text{Predicted} )^2$$
    - RMSE = $$\sqrt{\frac{1}{n} \sum ( \text{Actual} - \text{Predicted} )^2}$$
    - R² = 1 - $$\frac{\sum ( \text{Actual} - \text{Predicted} )^2}{\sum ( \text{Actual} - \bar{\text{Actual}} )^2}$$
- Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC
    - True positive: Predicted positive and actual's positive
    - False positive: Predicted positive but actual's negative
    - True negative: Predicted negative and actual's negative
    - False negative: Predicted negative but actual's positive
    - Precision = TP / (TP + FP) >>> how precise positive predictions are
    - Recall = TP / (TP + FN) >>> how many actual positives were captured
    - F1-score = 2 * (Precision * Recall) / (Precision + Recall) >>> balance precision and recall for class imbalance scenarios
    - Accuracy = (TP + TN) / (TP + TN + FP + FN) >>> overall correctness of the model   
    - Specificity = TN / (TN + FP) >>> how many actual negatives were captured
    - Precision-recall curve: y-axis = precision, x-axis = recall, area under curve (AUC) indicates model performance, higher is better.Good for imbalanced datasets.
    - AUC-ROC curve: y-axis = TPR (recall), x-axis = FPR (1 - specificity), area under curve (AUC) indicates model performance, higher is better. Good for balanced datasets.
- Times series: Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (sMAPE)
    - MAPE = $$\frac{1}{n} \sum \left|\frac{\text{Actual} - \text{Forecast}}{\text{Actual}}\right| \times 100$$
    - sMAPE = $$\frac{1}{n} \sum \frac{|\text{Forecast} - \text{Actual}|}{(|\text{Actual}| + |\text{Forecast}|)/2} \times 100$$
---
# ML Models
---
## K-Nearest Neighbors (KNN)
- instance-based learning algorithm
- classification and regression
- Predicts the class or value based on the majority class or average of the k-nearest neighbors in the feature space
- Distance metrics: Euclidean, Manhattan, Minkowski
Pros:
- Simple to implement and understand
- No training phase, making it fast for small datasets
Cons:
- Computationally expensive for large datasets
- Sensitive to irrelevant features and the choice of k

k is the hyperparameter representing the number of nearest neighbors to consider when making predictions.
- small k means more flexible decision boundary but may lead to overfitting
- large k means smoother decision boundary but may lead to underfitting

## Linear Regression
- regression algorithm
- Assumes linear relationship between input features and target variable
- Objective: minimize Mean Squared Error (MSE)
- Equation:  \( y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon \)
- Coefficients (β) represent the change in the target variable for a one-unit change
in the feature, holding other features constant.
Pros:
- Simple to implement and interpret
- Computationally efficient
Cons:
- Assumes linearity, which may not hold in real-world data
- Sensitive to outliers
Assumptions:
- Linearity: relationship between features and target is linear
- Independence: observations are independent of each other
- Homoscedasticity: constant variance of errors
- Normality: errors are normally distributed

## Logistic Regression
- classificaiton algorithm
- likelihood of a binary outcome based on input features
- Uses the logistic (sigmoid) function to map predicted values to probabilities
- Equation:  \( P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}} \)
Pros:
- Outputs probabilities, useful for binary classification
- Easy to implement and interpret
Cons
- Assumes linear decision boundary  
- May struggle with complex relationships

## Naive Bayes
- Bayesian classification algorithm
- Based on Bayes' theorem with strong (naive) independence assumptions between features
- Types: Gaussian, Multinomial, Bernoulli
Pros:
- Fast and efficient for large datasets
- Performs well with high-dimensional data
Cons:
- Assumes feature independence, which may not hold in practice

## Decision Tree
- Classification and regression algorithm
- Leaf nodes: represent class labels (classification) or continuous values (regression) - bottom of tree
- Does not require feature scaling or normalization
- Classification
    - Entropy: foused on information gain
    - Gini Impurity: focused on minimizing misclassification
    - Gixi Index: impurity measure based on the probability of a random sample being incorrectly classified
- Regression
    - Predicts value instead of class in each node
    - Variance reduction along feature
- Decision boundary
    - piecewise linear
    - perpendicular to an axis
Pros:   
- Easy to interpret and visualize
- Handles both numerical and categorical data   
Cons:
- Prone to overfitting, especially with deep trees
- High variance: sensitive to small changes in data

### Deep dive
- Recursive partition: repeatly split data into subsets to make outcome more homogeneous
- Split value: predictor value that divides data into two groups

## Ensemble
### Random Forests
- Combine multiple decision trees to improve performance
- Bagging (Bootstrap Aggregating): trains each tree on a random subset of data with replacement
- Random feature selection: each split considers a random subset of features
- Reduces overfitting and variance compared to single decision tree
- What's random:
    - Random sampling of data points for each tree (bagging)
    - Random selection of features for each split
    - Each node is a random set of m features from the global set at each split
    - Each tree uses a subset of samples

### Boosting
- Adaboost, gradient boosting, xgboost
- Predict the residuals of previous trees/models
- Sequentially add models to correct errors of prior models
- Learning rate: controls contribution of each tree to final prediction
### XGBoost
- parallel processing at node level
- level-wise tree growth
- regularization to reduce overfitting
### LightGBM
- leaf-wise tree growth with depth limitation
- histogram-based splitting for faster training
### CatBoost
- handles categorical features natively by converting them into numerical representations
- ordered boosting to reduce overfitting

## SVM
- Classification but also can be used for regression (SVR)
- Finds optimal hyperplane that maximizes margin between classes
- Support vectors: data points closest to the hyperplane
- Kernel trick: transforms data into higher dimensions to make it linearly separable
- Common kernels: linear, polynomial, RBF, Gaussian
Pros:
- Effective in high-dimensional spaces - for more features
- Robust to overfitting, especially with proper regularization
Cons:
- Computationally intensive for large datasets
- Sensitive to choice of kernel and hyperparameters
---
# Deep Learning
Using neural networks with multiple layers to learn complex patterns in data.
### Neural network
Main components:
- Layers: input, hidden, output
- Parameters: weights, biases
- Activation functions: ReLU, sigmoid, tanh
- Loss functions: MSE, cross-entropy
- Optimizers: SGD, Adam
Common models: CNNs, RNNs, Transformers, Autoencoders
backpropagation: algorithm used to train neural networks by minimizing the loss function through gradient descent.

single-layer perceptron (SLP)
- input layer, output layer
multi-layer perceptron (MLP) 
- input layer, hidden layers, output layer

Framework
Logic
- Forward pass > predictions
- Compute loss
- Backward pass > gradients
- Optimizer step > update parameters
- Zero gradients (to prevent accumulation or interference between batches, in plain language, clear old gradients before computing new ones)
- Evaluation on validation set (metrics)
---
# Reinforcement Learning
Learning via interaction with an environment to maximize cumulative reward.

Core idea: Agent learns policy π(a|s) that maps state → action to maximize expected return

Main components:
- Agent, environment
- State (s), action (a), reward (r), next state (s′)
- Policy π, value function V(s), Q-function Q(s, a)

Training loop:
- Observe state
- Choose action
- Receive reward, new state
- Update policy/value via reward feedback

Pros: Learns optimal sequential decisions

Cons: Slow convergence, unstable training, exploration–exploitation trade-off

Common algorithms: Q-learning, DQN, Policy Gradient, PPO, A3C
