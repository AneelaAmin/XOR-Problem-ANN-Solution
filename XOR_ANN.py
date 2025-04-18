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
