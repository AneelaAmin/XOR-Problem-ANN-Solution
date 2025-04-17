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
Achieves ~100% training accuracy using binary cross-entropy loss.
Converges in ~10,000 epochs with near-zero loss.
Matches XOR truth table accurately:

## 📈 Output
Input: [0, 1] → Predicted: 0.98 Input: [1, 1] → Predicted: 0.01
<img width="202" alt="image" src="https://github.com/user-attachments/assets/306cc79d-9a33-4326-b33c-d925f5a9ce74" />


## 📚 References
XOR Problem – Wikipedia
Andrew Ng’s Deep Learning Specialization – Week 3: Hidden Layers & Activation Functions
McCulloch-Pitts Neuron Model (1943)
MIT OpenCourseWare – Perceptron and MLP Architectures
