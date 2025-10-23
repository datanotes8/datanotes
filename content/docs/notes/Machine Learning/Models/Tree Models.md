---
title: "Tree Models"
draft: false
---

# Decision Trees
## Theory
- root node: depth 0 at top of tree
- split nodes: nodes with children, based on splits
- left nodes: nodes without children, terminal nodes



### Gini impurity
- Pure (gini=0) if all instances applies to one class
- Impure (gini=0.5) if instances are equally distributed among classes
- Training algorithm computes the Gini impurity Gi of the ith node
- The range of Gini impurity is [0, 0.5] for binary classification and [0, 1 - 1/K] for K classes.

$$
Gini(p) = 1 - \sum_{k=1}^{K} p_k^2
$$
where:
Gi is the Gini impurity of node i
pi,k is the proportion of class k instances in node i

### Entropy
- Alternative to Gini impurity for DecisionTreeClassifier
- Impurity measure based on information theory
- The range of Entropy is [0, log2(K)] for K classes.

$$
Entropy(p) = - \sum_{k=1}^{K} p_k \log_2(p_k)
$$

## Application
- Classification and regression
- No feature scaling or centering needed
- Makes little assumptions about the data
- Can also estimate the probability of an instance that belongs to a class 
- Parameters
    - max_depth: maximum depth of the tree
    - min_samples_split: minimum number of samples required to split an internal node
    - min_samples_leaf: minimum number of samples required to be at a leaf node
    - max_features: number of features evaluated at splitting a node
    - criterion: function to measure the quality of a split (default="gini" for classification, "squared_error" for regression)
    - ccp_alpha: minimal cost-complexity pruning


## Coding Example
```python
from sklearn.tree import DecisionTreeClassifier
iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

```

