# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Update: 1/16/26
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

# -------------------------
# Reproducibility
# -------------------------
# Fixing the random seeds ensures that:
# - The synthetic dataset is generated identically on every run
# - Weight initialization starts from the same parameters
# - Training dynamics (loss and accuracy curves) are repeatable
#
# This is critical for debugging, validating learning behavior,
# and demonstrating that observed improvements come from the
# optimization process itself—not from random chance.
np.random.seed(42)
random.seed(42)



batch_size = 1000
output_dim = 3
input_dim = 3
hidden_dim = 16     # 16 neurons
epochs = 100
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
raw_data = np.zeros((batch_size,input_dim), dtype=np.float32)
    
for i in range(batch_size):
    raw_data[i][0] = random_height()
    raw_data[i][1] = random_weight()
    raw_data[i][2] = random_foot_size()

# Used X to normalize the data that was generated.
# This is used later in our training loop; not locally
X = raw_data.copy()
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
# Set up our targets, should be a (1000, 3) with outputs
# [1,0,0] -> Large Animal
# [0,1,0] -> Medium Animal
# [0,0,1] -> Small Animal
targets = np.zeros((batch_size,output_dim), dtype=np.float32)

for j in range(batch_size):
    # Random size calculation but is completely standardized
    # to all the data for consistency. THIS IS FABRICATED
    size = raw_data[j][0] * raw_data[j][1] + raw_data[j][2]

    if size < 1500:
        targets[j] = [0, 0, 1]      # small

    elif size < 4500:
        targets[j] = [0, 1, 0]      # medium
    
    else:
        targets[j] = [1, 0, 0]      # large

# -------------------------
# Train / Test Split
# -------------------------
# Even with synthetic data, separating training and evaluation
# demonstrates that the network is learning general patterns
# rather than memorizing a fixed batch.
indices = np.random.permutation(batch_size)
split_idx = int(0.8 * batch_size)

train_idx = indices[:split_idx]
test_idx  = indices[split_idx:]

X_train, y_train = X[train_idx], targets[train_idx]
X_test,  y_test  = X[test_idx],  targets[test_idx]



# Time to write our Activation Function Helper function.
# For this we are using ReLU f(x) = max(0,x)
# ReLU is a standard choice as an activation function because
# it fixes the issue of vanishing gradients and is good with
# large computational problems

def relu(x):
    return np.maximum(0, x)


# This Helper function here is to help compute our loss:
# ** We use cross-entropy here ** 
# I really struggled trying to reason about implementing
# the formula for cross-entropy, so Chat-GPT 5.2 did help
# me debug my initial effort of:
#   loss = -np.log(np.sum(targets * probs))
def compute_loss(probs, targets):
    eps = 1e-9  # prevent log(0)

    correct_class_probs = np.sum(targets * probs, axis=1)
    correct_class_probs = np.clip(correct_class_probs, eps, 1.0)

    loss = -np.mean(np.log(correct_class_probs))
    return loss

# Need to define our weights shape and bias shape
# Earlier we defined:
#   input_dim = 3
#   output_dim = 3
#   hidden_dim = 16
# We have 5 layers.
#   Layer 1: w1 shape(3, 16) b1 shape(16,)
#   Layer 2: w2 shape(16, 16) b2 shape(16,)
#   Layer 3: w3 shape(16, 16) b3 shape(16,)
#   Layer 4: w4 shape(16, 16) b4 shape(16,)
#   Layer 5: w5 shape(16, 3) b5 shape(3,)


# -------------------------
# Weight Initialization (He Initialization)
# -------------------------
# This network uses ReLU activations, which zero-out negative values.
# Without proper scaling, signals can shrink or explode as they pass
# through multiple layers, making deep networks difficult to train.
#
# He initialization scales weights by sqrt(2 / fan_in), preserving
# activation variance across layers and maintaining healthy gradient
# flow during backpropagation.
#
# This choice is especially important for deeper ReLU networks and
# allows stable training without relying on framework-level defaults.


# w1 shape(3, 16)
weight1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)

# w2 shape(16, 16)
weight2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)

# w3 shape(16, 16)
weight3 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)

# w4 shape(16, 16)
weight4 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)

# w5 shape(16, 3)
weight5 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)

# b1 shape(16,)
bias1 = np.zeros((hidden_dim,), dtype=np.float32)

# b2 shape(16,)
bias2 = np.zeros((hidden_dim,), dtype=np.float32)

# b3 shape(16,)
bias3 = np.zeros((hidden_dim,), dtype=np.float32)

# b4 shape(16,)
bias4 = np.zeros((hidden_dim,), dtype=np.float32)

# b5 shape(3,)
bias5 = np.zeros((output_dim,), dtype=np.float32)

for epoch in range(epochs):

    # General Formula for forward pass:
    #    f = Wx + b
    #    output = data @ weights + bias
    # Activation Function:
    #    f(x) = max(0, x) // We have a helper function for this
    # We have 5 layers; time to iterate

    Z1 = X_train @ weight1 + bias1
    A1 = relu(Z1)

    Z2 = A1 @ weight2 + bias2
    A2 = relu(Z2)

    Z3 = A2 @ weight3 + bias3
    A3 = relu(Z3) 

    Z4 = A3 @ weight4 + bias4
    A4 = relu(Z4)

    logits = A4 @ weight5 + bias5

    # Time to manually do softmax, we have logits which are
    # our raw output data. Softmax just turns these into 
    # probabilites!
    # logits.shape(1000, 3)
    # We want softmax or probs.shape(1000, 3)
    #
    # Softmax manually is 5 steps:
    #   1. Find row-wise max:
    #       max_logits = max(logits, axis=1, keepdims=True)
    #   2. Shift logits:
    #       shifted = logits - max_logits
    #   3. Exponentiate:
    #       exp_scores = exp(shifted)
    #   4. Sum per row:
    #       sum_exp = sum(exp_scores, axis=1, keepdims=True)
    #   5. Normalize:
    #       probs = exp_scores / sum_exp

    # Here's the implementation:

    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_scores = np.exp(shifted)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)

    probs = exp_scores / sum_exp

    # -----------------------
    # Accuracy (diagnostic)
    # -----------------------
    pred = np.argmax(probs, axis=1)
    true = np.argmax(y_train, axis=1)
    acc = np.mean(pred == true)

    # Find our current loss on this pass:
    current_loss = compute_loss(probs, y_train)

    # Time to reweigh the gradients now!
    # =========================
    # Backpropagation (Core Idea)
    # =========================
    #
    # Backpropagation is NOT magic.
    # It is simply the chain rule applied repeatedly to the forward pass.
    #
    # Forward pass (simplified):
    #   X → Z1 → A1 → Z2 → A2 → Z3 → A3 → Z4 → A4 → logits → softmax → loss
    #
    # During backpropagation, we reverse this flow.
    #
    # The full derivative looks intimidating when expanded:
    #
    #   dL/dW1 = dL/dlogits · dlogits/dA4 · dA4/dZ4 · dZ4/dA3 · ...
    #            · dA1/dZ1 · dZ1/dW1
    #
    # BUT the key realization is:
    #
    #   - Each layer has the SAME local structure
    #   - Matrix multiplication bundles thousands of partial derivatives
    #   - NumPy handles the summation automatically
    #
    # So instead of writing massive equations, we compute:
    #
    #   1. Error signal at the output
    #   2. Propagate it backward through each layer
    #   3. Compute gradients using matrix multiplications
    #
    # This compression is why deep learning scales.
    # The math is the same — it’s just vectorized.
    #
    # NOTE:
    # This section was developed with guided assistance from ChatGPT.
    # Backpropagation in condensed, vectorized form was by far the most
    # challenging part of this project to implement correctly.
    #
    # I first derived the full expanded equations by hand to understand
    # the chain rule per layer, then used guided help to translate those
    # equations into correct, shape-consistent NumPy operations.
    #
    # The final implementation reflects my understanding of:
    # - gradient flow through dense layers
    # - ReLU masking
    # - softmax + cross-entropy simplification
    # - batch-wise matrix calculus
    #
    # This experience made it clear why modern frameworks (e.g. PyTorch)
    # abstract autograd — not because the math is trivial, but because
    # it is extremely error-prone to implement correctly at scale.


    # -----------------------
    # Backprop (vectorized)
    # -----------------------
    N = X_train.shape[0]                    # batch_size

    # 1) softmax + cross-entropy gradient
    dlogits = (probs - y_train) / N         # (N, 3)

    # 2) Layer 5 (A4 -> logits)
    dW5 = A4.T @ dlogits                     # (16, N) @ (N, 3) = (16, 3)
    db5 = np.sum(dlogits, axis=0)            # (3,)
    dA4 = dlogits @ weight5.T                # (N, 3) @ (3, 16) = (N, 16)

    # 3) ReLU back through Layer 4
    dZ4 = dA4 * (Z4 > 0)                     # (N, 16)

    dW4 = A3.T @ dZ4                         # (16, 16)
    db4 = np.sum(dZ4, axis=0)                # (16,)
    dA3 = dZ4 @ weight4.T                    # (N, 16)

    # 4) ReLU back through Layer 3
    dZ3 = dA3 * (Z3 > 0)

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)
    dA2 = dZ3 @ weight3.T

    # 5) ReLU back through Layer 2
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ weight2.T

    # 6) ReLU back through Layer 1
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X_train.T @ dZ1                    # (3, N) @ (N, 16) = (3, 16)
    db1 = np.sum(dZ1, axis=0)                # (16,)

    # -----------------------
    # Gradient step (SGD)
    # -----------------------
    weight5 -= learning_rate * dW5
    bias5   -= learning_rate * db5

    weight4 -= learning_rate * dW4
    bias4   -= learning_rate * db4

    weight3 -= learning_rate * dW3
    bias3   -= learning_rate * db3

    weight2 -= learning_rate * dW2
    bias2   -= learning_rate * db2

    weight1 -= learning_rate * dW1
    bias1   -= learning_rate * db1

    # Note: I used guided help (ChatGPT) to validate the train/test evaluation addition.
    # The core goal is to show measurable generalization, not just decreasing loss.

    # -----------------------
    # Test set accuracy
    # -----------------------
    Z1_t = X_test @ weight1 + bias1
    A1_t = relu(Z1_t)
    Z2_t = A1_t @ weight2 + bias2
    A2_t = relu(Z2_t)
    Z3_t = A2_t @ weight3 + bias3
    A3_t = relu(Z3_t)
    Z4_t = A3_t @ weight4 + bias4
    A4_t = relu(Z4_t)
    logits_t = A4_t @ weight5 + bias5

    max_logits_t = np.max(logits_t, axis=1, keepdims=True)
    exp_scores_t = np.exp(logits_t - max_logits_t)
    probs_t = exp_scores_t / np.sum(exp_scores_t, axis=1, keepdims=True)

    test_pred = np.argmax(probs_t, axis=1)
    test_true = np.argmax(y_test, axis=1)
    test_acc = np.mean(test_pred == test_true)

    print(f"Epoch {epoch} | Loss: {current_loss:.4f} | Train Acc: {acc:.3f} | Test Acc: {test_acc:.3f}")





                    




