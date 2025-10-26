---
title: "Deep Learning"
draft: false
---
# Deep Learning
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