---
title: "Neural Networks and Deep Learning"
draft: false
---
# Neural Networks and Deep Learning
## Theory
### NN Elements 
Neural network is a general framework that uses layers of interconnected nodes (neurons) to learn complex patterns from data. Deep learning is a subset of NN and refers to neural networks with multiple hidden layers that can learn hierarchical representations.
Neural network consists of nodes and conenctions, where they are connected by weights or parameters. These weights are learned in a process called backpropagation.

{{< hint info >}} 
Nodes
{{< /hint >}}
Input nodes, hidden nodes, output nodes
- Input layers are the feature vectors x
- Hidden layers are nodes between IO nodes that learns representations from the data
- Ouput layer gives the predictions y

{{< hint info >}} 
Activation functions
{{< /hint >}}
Functions used to help the network learn. 
- Softmax, ReLu, Sigmoid, Tanh   

We use activation function to get the x-axis and y-axis values for the input data. The y-axis value depends on the activation function chosen.

{{< hint info >}} 
Weights and biases
{{< /hint >}}
Weights control importance of input features
Biases allow model to fit data better by shifting activation function.

{{< hint info >}} 
Backpropagation
{{< /hint >}}
Fiting the data to the neural network and provide parameters/weights for the connections between nodes.

{{< hint info >}} 
Epoch and batching
{{< /hint >}}
Epoch: one full pass through training data
Batch: subset of data used for one gradient update

### perceptron
{{< hint info >}} 
perceptron
{{< /hint >}}
This is the simplest type of neural network, consisting of a single layer of output nodes connected to input features. It can only learn linear decision boundaries.

### Attention
Computes weighted average of all input vectors. This allows each token to look at other tokens for context. Each token gets:
- Query (Q): goal
- Key (K): match
- Value (V): information
$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

### Transformers
Type of neural network for sequences such as text or time series, but they don't use recurrence RNN or convolutions CNN.
Input embeddings
- Convert token into vectors.
Positional encodings
- Add sequence order info because transformers have no built-in order like RNNs.
Self-attention mechanism
- Learns the section of the input to focus on for each token.
Feed-forward layer
- NN applied to each position's output.
Layer normalization and residuals
- Stabilizes and speed up training


## Application
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