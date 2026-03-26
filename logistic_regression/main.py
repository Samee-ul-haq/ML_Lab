# ------------ Using NUMPY and MATPLOTLIB ---------
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.datasets import load_diabetes
import math

dataset=load_diabetes()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['daibetic']=dataset.target

X=df.drop(columns='daibetic')
y=df['daibetic']

train_size=int(0.8*len(X))
np.random.seed(42)
indicies=np.random.permutation(len(X))

train_indicies=indicies[:train_size]
test_indicies=indicies[train_size:]

X_train=X.iloc[train_indicies]
X_test=X.iloc[test_indicies]
y_train=y.iloc[train_indicies]
y_test=y.iloc[test_indicies]

def sgd(X,y,learning_rate,epochs,batch_size):
    X=np.array(X)
    y=np.array(y).reshape(-1, 1)
    m=len(X)
    theta=np.random.randn(11,1)
    X_bias=np.c_[np.ones((m,1)),X]

    cost_history=[]
    for epoch in range(epochs):
        indxs=np.random.permutation(m)
        X_shuffled=X_bias[indxs]
        y_shuffled=y[indxs]

        for i in range(0,m,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            y_batch=y_shuffled[i:i+batch_size]

            gradients=2/batch_size*\
                X_batch.T.dot(X_batch.dot(theta)-y_batch)
            theta=theta-learning_rate*gradients

        predictions=X_bias.dot(theta)
        cost=np.mean((predictions-y)**2)
        cost_history.append(math.sqrt(cost))

        if epoch%100==0:
            print(f"Epoch {epoch},Cost:{math.sqrt(cost)}")

    return theta,cost_history

theta_final,cost_history=sgd(X_train,y_train,learning_rate=0.1,epochs=1000,batch_size=64)

import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function during Training')
plt.show()

print(f"Final parameters: {theta_final}")

X_test=np.array(X_test)
y_test=np.array(y_test).reshape(-1,1)
m=len(X_test)
X_test_bias=np.c_[np.ones((m,1)), X_test]
y_pred=X_test_bias.dot(theta_final)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)

print(f"loss:{y_test-y_pred}")