---
title: "My Hidden Page"
draft: false
---

# Quantile Regression

## Resources
> [<span style="display: none">Quantile Regression Paper (2001)</span>](https://www.aeaweb.org/articles?id=10.1257/jep.15.4.143)  
[<span style="display: none">Scikit-learn Tutorial</span>](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html)  
[<span style="display: none">Xgboost Implementation</span>](https://xgboost.readthedocs.io/en/stable/python/examples/quantile_regression.html)



## Theory
Quantile regression is slightly different from general regression or OLS; it is a special model that helps calculate different quantile levels. The underlying theory is to estimate the conditional distribution. This is particularly useful when you need to compute a range. Especially in cases where there are specific business metric constraints on the model, quantile regression can help achieve the goal. It can penalize both over-estimation and under-estimation, depending on the situation.

What differentiates it from regular OLS is the use of quantile loss, or pinball loss. The loss function in general models focuses on minimizing MAE or MSE, whereas this one focuses on ensuring that enough of the predicted portion is greater than the label. By looking at this loss function, we can analyze the model's logic.


$$
L_q(y, \hat{y}) = 
\begin{cases} 
q \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\\\
(1 - q) \cdot (\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

- The $(y - \hat{y})$ is the error or residual, true value minus the predicted value.
- $q$ is also referred to as $\alpha$, and it is the most important parameter that determines the quantile level. For example, 0.9 represents the 90th percentile.
- If the predicted value is smaller than the true value, meaning actual - prediction > 0, then the penalty is $q$ times the error, which is called under-estimation.
- If the predicted value is larger than the true value, meaning actual - prediction < 0, then the penalty is $(1 - q)$ times the error, which is called over-estimation.
- The model minimizes the loss, and the value of $\alpha$ determines whether it biases towards predicting higher or lower values.

## Example
- For example, when quantile = 0.9:
    - Prediction = 50, True value = 100, penalty = 0.9 × (100 - 50) = 45
    - Prediction = 150, True value = 100, penalty = (1 - 0.9) × (150 - 100) = 5
    - For instance, if Instacart requires 90% of deliveries to be on-time, you can use quantile 0.9 for prediction.
- For example, when quantile = 0.1:
    - Prediction = 50, True value = 100, penalty = 0.1 × (100 - 50) = 5
    - Prediction = 150, True value = 100, penalty = (1 - 0.1) × (150 - 100) = 45
    - For instance, an e-commerce company wants to calculate the minimum stock level to avoid over-predicting inventory, thus reducing inventory costs.
- The advantage of quantile regression is that it's not just a single regression; it can be used in various models, such as random forest, xgboost, and neural networks. Some of these are even built-in and easy to implement. For example, using XGBoost.


## Coding Example
```python
import xgboost as xgb
params = {
    "objective": "reg:quantile", 
    "alpha": 0.9,                 
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 100
}
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)
y_pred_90th = model.predict(X_test)
```

