# 🌸 MatrixFlowers

A lightweight deep learning framework built from scratch in Python, with a custom matrix backend in C.

Inspired by PyTorch and TensorFlow, MatrixFlowers implements dynamic autograd (reverse-mode automatic differentiation), a Keras-style layer API, and a TensorFlow-style `GradientTape` — all without depending on PyTorch or TensorFlow.

> ⚠️ **Experimental / learning project.** Not intended for production use.

---

## Features

- **Custom autograd engine** — dynamic computation graph with reverse-mode autodiff (`tensor.py`)
- **C matrix backend** — element-wise ops and matrix multiplication via `ctypes` + a compiled `lib.so` (with optional OpenBLAS)
- **Keras-style layers** — `Dense`, `Dropout`, `Sequential` with `.summary()`
- **GradientTape API** — TensorFlow-style gradient tracking
- **Optimizers** — `SGD` (with momentum) and `Adam` (with weight decay)
- **Loss functions** — MSE, Cross-Entropy, Binary Cross-Entropy, Huber
- **Activations** — ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU, ELU, and `manu` (a custom `x * sin(x)` activation)
- **High-level Trainer** — `.fit()`, `.evaluate()`, `.predict()`

---

## Project Structure

```
matrixflowers/
├── tensor.py                        # Tensor class + autograd engine
├── nn/
│   ├── layers.py                    # Dense, Dropout, Sequential
│   └── activations.py               # Activation functions (autograd-compatible)
├── train/
│   ├── tape.py                      # GradientTape
│   ├── optimizers.py                # SGD, Adam (Tensor-based)
│   ├── losses.py                    # MSE, CrossEntropy, BCE, Huber
│   └── trainer.py                   # High-level Trainer
├── type/matrix/
│   ├── py_implementation/matrix.py  # Matrix class (ctypes wrapper)
│   └── helps-in-c/operations.c      # C backend for matrix ops
└── model/                           # Legacy parameter-based API
```

---

## Quick Start

### Regressão com Trainer (alto nível)

```python
import numpy as np
from tensor import Tensor
from nn.layers import Sequential, Dense
from train.trainer import Trainer
from train.optimizers import Adam
from train.losses import mse

# Dataset: f(x) = x²
X = np.linspace(-2, 2, 200).reshape(-1, 1).astype(np.float32)
y = (X ** 2).astype(np.float32)

# Modelo
model = Sequential([
    Dense(16, input_dim=1,  activation="relu"),
    Dense(16, input_dim=16, activation="relu"),
    Dense(1,  input_dim=16),
])

model.summary()

# Treino
optimizer = Adam(model.trainable_tensors(), lr=0.01)
trainer = Trainer(model=model, optimizer=optimizer, loss_fn=mse)
trainer.fit(X, y, epochs=500, batch_size=32, verbose=True)
```

### Treino manual com GradientTape

```python
from train.tape import GradientTape

for epoch in range(500):
    x = Tensor(X, requires_grad=False)
    y_t = Tensor(y, requires_grad=False)

    with GradientTape() as tape:
        pred = model(x)
        loss = mse(pred, y_t)

    grads = tape.gradient(loss, model.trainable_tensors())

    for g, v in zip(grads, model.trainable_tensors()):
        v.grad = g.data if g is not None else None

    optimizer.step()
    optimizer.zero_grad()
```

---

## Examples

| File | Description |
|---|---|
| `example-quadratic.py` | Learns `f(x) = x²` |
| `example-sin.py` | Learns `f(x) = sin(x)` |
| `example.py` | General regression demo |
| `perceptron-experiments/` | Early perceptron experiments |

---

## Activations

| Name | Formula |
|---|---|
| `relu` | `max(0, x)` |
| `sigmoid` | `1 / (1 + e⁻ˣ)` |
| `tanh` | `tanh(x)` |
| `softmax` | stable softmax |
| `leaky_relu` | `x if x > 0 else 0.01x` |
| `elu` | `x if x > 0 else α(eˣ - 1)` |
| `manu` | `x * sin(x)` *(custom)* |

---

## C Backend

Matrix operations (add, sub, mul, div, matmul) are implemented in C and loaded via `ctypes`:

```bash
# Compile manually
cd type/matrix/helps-in-c
bash compile.sh
```

The `matmul` uses OpenBLAS when available for faster matrix multiplication.

---

## Requirements

```
numpy
```

The C backend requires GCC (and optionally OpenBLAS) to compile `lib.so`.

---

## Status

Work in progress. Core autograd, layers, and training loop are functional. APIs may change.
