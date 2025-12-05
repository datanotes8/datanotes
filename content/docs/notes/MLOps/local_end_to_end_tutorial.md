
---
title: "End-to-End Deployment Tutorial"
draft: false
---

# Simple Local end-to-end deployment tutorial.
This is a quick demonstration of an end-to-end machine learning deployment pipeline on a local machine. There are a few key steps involved: data processing, model training, model serialization, model serving, and front-end UI. The end result is a front-end UI that allows user to input values for the features and get a real-time prediction from the REST API. This would be considered a barebones MLOps deployment pipeline on a local machine, but captures the key steps involved in a real-world deployment.

Recommended file structure:
```
ml_e2e_project/
├── main.py
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── generate_data.py
│   ├── preprocess_data.py
│   ├── model_training.py
│   ├── model_serialization.py
│   ├── serve_fastapi.py
└── └── ui_streamlit.py
```
Required libraries:
pandas numpy scikit-learn torch onnx onnxruntime fastapi uvicorn streamlit


## Generate Synthetic Data (generate_data.py)
```python
import pandas as pd
import numpy as np
import os

def generate_data(n=500):
    """
    Generate synthetic train/test CSVs under project_root/data
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(base_dir)               # root/
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Create synthetic data
    np.random.seed(42)
    X1 = np.random.randn(n)
    X2 = np.random.rand(n) * 10
    X3 = np.random.choice(["A", "B", "C"], size=n)
    y = 3 * X1 + 0.5 * X2 + (X3 == "B") * 2 + np.random.randn(n) * 0.3

    df = pd.DataFrame({"x1": X1, "x2": X2, "x3": X3, "target": y})
    train = df.sample(frac=0.8, random_state=1)
    test = df.drop(train.index)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return train_path, test_path
```

## Data Preprocessing (preprocess_data.py)
```python
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(train_path):
    """
    Reads CSV, fits OneHotEncoder + StandardScaler, returns X, y, enc, scaler
    """
    df_train = pd.read_csv(train_path)

    cat = df_train[["x3"]]
    num = df_train[["x1", "x2"]]

    enc = OneHotEncoder(sparse_output=False).fit(cat)
    scaler = StandardScaler().fit(num)

    X_cat = enc.transform(cat)
    X_num = scaler.transform(num)
    X = np.hstack([X_num, X_cat])
    y = df_train["target"].values

    return X, y, enc, scaler
```

## Model Training (model_training.py)
```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SimpleNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model(X, y, enc, scaler, epochs=20, batch_size=32, lr=0.01):
    """
    Train a simple feedforward NN on preprocessed data.
    Returns trained model along with enc and scaler.
    """
    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SimpleNN(in_dim=X.shape[1])
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Save model + preprocessing
    torch.save({"model": model.state_dict(), "enc": enc, "scaler": scaler}, "model.pt")

    return model, enc, scaler
```

## Model Serialization to ONNX (model_serialization.py)
```python
import torch
import onnx

def export_onnx(model, X_shape, path="model.onnx"):
    dummy_input = torch.randn(1, X_shape[1])
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"]
    )
    onnx.checker.check_model(path)
    return path
```

## Model Serving with FastAPI (serve_fastapi.py)
```python
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
sess = ort.InferenceSession("model.onnx")

@app.post("/predict")
def predict(data: dict):
    X = np.array(data["features"], dtype=np.float32)
    pred = sess.run(None, {"input": X})[0]
    return {"prediction": pred.tolist()}
```

## Front end UI (ui_streamlit.py)
```python
import streamlit as st
import requests
import numpy as np

st.title("E2E ML Demo")

x1 = st.number_input("x1", value=0.0)
x2 = st.number_input("x2", value=5.0)
x3 = st.selectbox("x3", ["A", "B", "C"])

cat_map = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]}
features = np.array([[x1, x2] + cat_map[x3]])

if st.button("Predict"):
    res = requests.post("http://127.0.0.1:8000/predict", json={"features": features.tolist()})
    st.write("Prediction:", res.json()["prediction"])
```

## Main Script (main.py)
```python
import logging
import sys
import os
import subprocess
import time
import signal

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------- Add src/ to Python path ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # root/
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

# ---------- Import modules ----------
from generate_data import generate_data
from preprocess_data import preprocess_data
from model_training import train_model
from model_serialization import export_onnx
from serve_fastapi import app  # for serving instructions

# ---------- Pipeline ----------
logger.info("Step 1: Generate data")
train_path, test_path = generate_data()
logger.info("Data generated at %s", train_path)

logger.info("Step 2: Preprocess data")
X, y, enc, scaler = preprocess_data(train_path)
logger.info("Preprocessing done. Shapes: X=%s, y=%s", X.shape, y.shape)

logger.info("Step 3: Train model")
model, enc, scaler = train_model(X, y, enc, scaler)
logger.info("Model training done.")

logger.info("Step 4: Serialize Model Object to ONNX")
onnx_path = export_onnx(model, X.shape)
logger.info("Model exported to %s", onnx_path)

# ---------- Step 5: Start FastAPI ----------
logger.info("Step 5: Start FastAPI server")
# Launch FastAPI in a separate process
fastapi_proc = subprocess.Popen(
    ["uvicorn", "src.serve_fastapi:app", "--reload", "--port", "8000"]
)
logging.info("Waiting 1 minute for FastAPI to start...")
time.sleep(60)  # wait 60 seconds

# ---------- Step 6: Launch Streamlit UI ----------
logger.info("Step 6: Launch Streamlit")
# Launch Streamlit in a separate process
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "src/ui_streamlit.py"]
)

# ---------- Keep main.py alive, terminate subprocesses on Ctrl+C ----------
try:
    logger.info("E2E pipeline running. Press Ctrl+C to stop both servers.")
    # Wait indefinitely
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Ctrl+C detected. Terminating servers...")
    fastapi_proc.send_signal(signal.SIGINT)
    streamlit_proc.send_signal(signal.SIGINT)
    fastapi_proc.wait()
    streamlit_proc.wait()
    logger.info("Both servers terminated. Exiting.")
```