# XOR-Problem-ANN-Solution
Solving the XOR logic problem using an Artificial Neural Network (ANN) in Python with clear explanation and code.
# XOR Problem Solved with Artificial Neural Network

This project demonstrates how a multi-layer neural network can solve the classic XOR problem, which is not solvable by a single-layer perceptron.

## 🧠 What is the XOR Problem?

XOR (exclusive OR) outputs `1` only when the two binary inputs are different. It's a simple but classic example of a non-linearly separable problem.

| A | B | XOR |
|---|---|-----|
| 0 | 0 |  0  |
| 0 | 1 |  1  |
| 1 | 0 |  1  |
| 1 | 1 |  0  |

## 🤔 Why Use a Neural Network?

A single-layer perceptron fails at XOR due to its linear limitation. But a Multi-Layer Perceptron (MLP) introduces non-linearity through hidden layers and activation functions.

## 🧱 Network Architecture

- **Input Layer:** 2 neurons (A and B)
- **Hidden Layer:** 2 neurons with ReLU activation
- **Output Layer:** 1 neuron with sigmoid activation

## 🧾 Code Overview

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

model=Sequential()
model.add(Dense(2,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X,y,epochs=100,verbose=0)

# Evaluating the Accuracy
_,accuracy=model.evaluate(X,y)
print(f"Accuracy : {accuracy*100:.2f}%")

predictions=model.predict(X)
predictions=np.round(predictions).astype(int)

# Printing the Predicted Values and the Accurate Outputs as well
print("Prediction: ")
for i in range(len(X)):
    print(f"Input {X[i]} => Predicted Output: {predictions[i]} , Actual Output: {y[i]}")

## 🔧 How to Run

1. Clone the repo:
git clone https://github.com/yourusername/xor-problem-ann-solution.git
cd xor-problem-ann-solution

2. python XOR_ANN.py

## 📈 Accuracy
Achieves near 100% accuracy using binary cross-entropy loss.
Achieves ~100% training accuracy after 10,000 epochs.  
Loss converges to near zero, and predictions match XOR truth table.

# Output
Input: [0, 1] → Predicted: 0.98 Input: [1, 1] → Predicted: 0.01


## 📚 References
- [XOR Problem Wiki](https://en.wikipedia.org/wiki/Exclusive_or)
- Andrew Ng’s Deep Learning Specialization – Week 3 (Hidden Layers & Activation Functions)
- McCulloch-Pitts Neuron Model
- Perceptron and MLP Architectures (MIT OCW)

