# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Update: 12/23/25
# File Name: neural_network.py
#
# Description:
#   This project implements a multi-layer neural network from scratch
#   using NumPy, extending a single-layer model to include hidden layers
#   and nonlinear activation functions.
#
#   The goal of this project is to demonstrate how depth and nonlinearity
#   increase a model’s expressive power while preserving the same
#   fundamental training loop: forward pass, loss computation,
#   parameter optimization, and iteration over epochs.
#
#   No machine learning frameworks are used. Every component—from data
#   flow to parameter updates—is implemented explicitly to reinforce a
#   first-principles understanding of how modern neural networks learn.

# =========================
# Inference (Forward Pass)
# =========================
# 1. Data (batch)
#    X shape: (batch_size, input_dim)
# 2. Layer 1 (Input → Hidden 1)
#    Z1 = X @ W1 + b1
#    A1 = f(Z1)
# 3. Layer 2 (Hidden 1 → Hidden 2)
#    Z2 = A1 @ W2 + b2
#    A2 = f(Z2)
# 4. Layer 3 (Hidden 2 → Hidden 3)
#    Z3 = A2 @ W3 + b3
#    A3 = f(Z3)
# 5. Layer 4 (Hidden 3 → Hidden 4)
#    Z4 = A3 @ W4 + b4
#    A4 = f(Z4)
# 6. Output Layer (Hidden 4 → Output)
#    logits = A4 @ W5 + b5
# 7. Prediction step
#    - Regression: y_hat = logits
#    - Classification: probs = softmax(logits)
# =========================
# Training Loop
# =========================
# Outer loop: for epoch in epochs
#   Inner loop: for batch in batches
#
# 8. Forward pass (steps 1–6)
# 9. Loss computation (from logits + targets)
# 10. Backpropagation (gradients via chain rule)
#     - dL/dW5, dL/db5
#     - propagate through activations
#     - dL/dW1, dL/db1
# 11. Optimization step (update all weights + biases)
# 12. Repeat for all batches → next epoch

# =========================
# Define the Problem
# =========================
# A 5-layer fully connected neural network that classifies
# animals as small, medium, or large based on weight,
# height, and foot size, using ReLU activations, softmax output,
# and cross-entropy loss, implemented entirely from scratch
# in NumPy.

import numpy as np
import random

batch_size = 1000
output_dim = 3
input_dim = 3
hidden_dim = 16     # 16 neurons
epochs = 10
epsilon = 1e-3 
learning_rate = 0.05


# Generates height in Meters(m)
def random_height():
    height = random.uniform(1, 6)
    return height

# Generates weight in Kilograms(kg)
def random_weight():
    weight = random.uniform(50, 1500)
    return weight

# Generates random foot_size(cm)
def random_foot_size():
    foot_size = random.uniform(5, 25)
    return foot_size


# Fill data with randomness
data = np.zeros((batch_size,input_dim), dtype=np.float32)
    
for i in range(batch_size):
    data[i][0] = random_height()
    data[i][1] = random_weight()
    data[i][2] = random_foot_size()
    