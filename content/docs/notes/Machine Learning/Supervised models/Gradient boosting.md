---
title: "My Hidden Page"
draft: true
---

# ML Notes

# Xgboost vs lightgbm

## Differences



## Gradient boosting
- Ensamble predictions of weak-learners to improve model performance
- Weak learns are sometimes slightly better than random guess
- Inital model trains on whole dataset
- Next iteration focus on the residuals of actual label and prediction

## Difference compared to Random Forest
- Random forest uses bagging
    - Bagging (Bootstrap Aggregating) is ensemble learning method where it uses bootstrap sampling to improve the model.
    - Random sampling with replacement
    - New training set has same number of data points as original, so duplicates exist (aka bootstrapping)
    - Benefit of bagging is reduce variance of the model, avoids overfitting
    - Downside of bagging is increase in bias, increase underfitting



## XGBoost (eXtreme Gradient Boosting)
- It is a scalable distributed gradient-boosted decision tree algorithm.
- Xgboost uses a depth-first tree growth method underneath the hood.
- Level-wise tree growht, grows downwards as trees increase
- Decision trees by nature predicts the label by a bunch of if-then-else true/false decisions.
## LightGBM 
- It is a histogram based learning algorithm that buckets values, hence less memory
- But also may overfit more than xgboost
- Lightgbm uses a leaf-wise tree growth method. This results in deeper trees.
- Grows horizontally as trees grow
- Highly efficient and optimized for paralle and distributed computing
- Less memory requirements due to the shallow trees.

## Simple terms
XGBoost: Grows trees deeper in a level-wise fashion.
LightGBM: Grows trees with fewer levels but more nodes in a leaf-wise fashion.
