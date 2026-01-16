# Neural Network from Scratch

A multi-layer neural network implementation built entirely with NumPy to understand the fundamentals of deep learning without abstraction.

## Overview

This project implements a 5-layer fully connected network that classifies synthetic animal data (height, weight, foot size) into three size categories. No ML frameworks are used—every component from forward propagation to backpropagation is written explicitly.

**Architecture:**
- Input: 3 features
- Hidden layers: 4 layers × 16 neurons (ReLU activation)
- Output: 3 classes (softmax + cross-entropy loss)
- Optimizer: Stochastic Gradient Descent

## Key Implementation Details

- **He initialization** for stable gradient flow through ReLU layers
- **Numerically stable softmax** (max-subtraction trick)
- **Train/test split** to validate generalization
- **Vectorized backpropagation** using matrix calculus

## Learning Notes

This is an early project in my machine learning journey. I built this to develop first-principles understanding of how neural networks actually work under the hood.

**The hardest part wasn't the math—it was reasoning about shapes.** Understanding how matrix dimensions flow through the network, why certain operations require transposes, and how batch processing affects gradient computation took significant effort. Tools like PyTorch abstract this away with autograd, but implementing it manually revealed why shape mismatches are such a common source of bugs.

Other challenging areas:
- **Backpropagation condensation:** Translating the expanded chain rule into vectorized NumPy operations required guided assistance. I first derived the math by hand, then worked through the implementation with help to ensure correctness.
- **Numerical stability:** Learning why softmax needs max-subtraction and loss functions need epsilon clipping.
- **Cross-entropy implementation:** My initial approach was incorrect; debugging this taught me to validate against known formulas rather than assume correctness.

The experience made it clear why frameworks exist—not because the underlying operations are conceptually difficult, but because getting the implementation details right at scale is genuinely hard.

## Requirements

```
numpy
```

## Usage

```bash
python neural_network.py
```

The network trains for 100 epochs and prints training/test accuracy per epoch.

## What's Next

This project is part of a personal ML curriculum focused on building foundational models from scratch. Next up:

1. **Single-head transformer** (attention mechanism, positional encoding)
2. **Multi-head transformer** (parallelized attention, full architecture)

The goal is to understand attention and sequential processing at the same first-principles level before moving to frameworks.

---

*Built as a learning exercise to understand neural networks from the ground up.*