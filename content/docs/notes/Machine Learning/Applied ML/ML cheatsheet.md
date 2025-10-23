---
title: "ML Cheatsheet"
draft: false
---
# Genearl ML Concepts
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

## Scaling and normalization
- Scaling: transforming features to a specific range, such as [0, 1] or [-1, 1]. Common methods include Min-Max Scaling and Standardization (Z-score normalization).
- Normalization: transforming features to have a mean of 0 and a standard deviation of 1. This is often done using StandardScaler in libraries like scikit-learn.

# ML Models
## Decision Tree
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

## Ensemble and Random Forests


## Graident Boosting



## DL 
Using neural networks with multiple layers to learn complex patterns in data.
### Neural network model
Main components:
- Layers: input, hidden, output
- Parameters: weights, biases
- Activation functions: ReLU, sigmoid, tanh
- Loss functions: MSE, cross-entropy
- Optimizers: SGD, Adam

Common models: CNNs, RNNs, Transformers, Autoencoders
backpropagation: algorithm used to train neural networks by minimizing the loss function through gradient descent.

multi-layer perceptron (MLP) 

Framework
Logic
- Forward pass > predictions
- Compute loss
- Backward pass > gradients
- Optimizer step > update parameters
- Zero gradients (to prevent accumulation or interference between batches, in plain language, clear old gradients before computing new ones)
- Evaluation on validation set (metrics)

- ETL
    -  import, preprocessing, batching for gradient updates, dataloader
- Model
    - input layer, hidden layers for representation learning, output layer for prediction
    - Parmeters: weights and biases
    - Activation functions: ReLU, sigmoid, tanh
    - Loss functions: MSE, cross-entropy
    - Forward pass equation:
        - z = W*x + b
        - a = activation(z)         
    - Types: feeedforward, CNN(spatial features), RNN/Transformer(sequence data)
- Loss function
    - Measures difference between predicted and true values
    - Regression: MSE
    - Classification: cross-entropy

- Optimization
    - Algorithms to minimize loss function by updating model parameters
    - Variants:
        - SGD: basic gradient descent
        - Adam: adaptive learning rates 
        - RMSProp
    - Hyperparameters: learning rate, batch size, epochs

- Backpropagation
    - Algorithm to compute gradients of loss w.r.t model parameters
    - Uses chain rule to propagate error backward through layers
    - Update rule: param = param - learning_rate * gradient
- Epochs and Batching
    - Epoch: one complete pass through the entire training dataset
    - Batch: subset of training data used to compute gradients and update parameters
    - Mini-batch gradient descent: balances efficiency and convergence by using small batches

Code example of a feed-forward neural network in PyTorch:
```python
import torch
import torch.nn as nn
import torch.optim as optim
# ===============================
# 1. Simple Feed-Forward Network
# ===============================
class FeedForwardNN(nn.Module):
    """
    Simple PyTorch NN module
    1st hidden layer: input_dim -> hidden_dim
    2nd hidden layer: hidden_dim -> hidden_dim
    Output layer: hidden_dim -> 1 (regression)
    Activation: ReLU
    Training loop: repeats forward -> loss -> backward for each epoch
    """
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim  : number of input features
        hidden_dim : neurons in hidden layers
        output_dim = 1 (single regression target)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Layer 1
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Layer 2
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)          # Output layer

    def forward(self, x):
        x = self.act1(self.fc1(x))   # first nonlinear transform
        x = self.act2(self.fc2(x))   # deeper representation
        out = self.fc3(x)            # scalar prediction
        return out

def data_loader(df,target_column,batch_size=32,shuffle=False):
    """
    Convert pandas DataFrame to PyTorch DataLoader
    
    Args:
        df: pandas DataFrame
        target_column: name of the target variable
        batch_size: size of each batch
        
    Returns:
        Batches of data in format of a tuple(features, labels)
        Features is tensor of shape (batch_size, num_features)
        Labels is tensor of shape (batch_size, 1)
    """
    dataset = CustomDataset(df, target_column)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader
# ===============================
# 2. Define model + loss + optimizer
# ===============================
model = FeedForwardNN(input_dim=10, hidden_dim=64)
criterion = nn.MSELoss() # mean squared error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer

# ===============================
# 3. Training loop
# ===============================
for epoch in range(10):
    # Example batch of data
    x = torch.randn(32, 10)  # 32 samples, 10 features
    y = torch.randn(32, 1)   # true numeric targets
    # Forward pass
    preds = model(x)
    # Compute loss
    loss = criterion(preds, y)
    # Backward + optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")
```

## RL
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