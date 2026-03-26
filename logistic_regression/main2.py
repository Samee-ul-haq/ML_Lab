# ------------ Building from scratch Using Pytorch and MATPLOTLIB -------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes

dataset=load_diabetes()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['diabetic']=dataset.target

X=df.drop(columns='diabetic') 
y=df['diabetic'] 
threshold=np.median(y)
y_binary=np.where(y>threshold,1,0)
y=pd.Series(y_binary)
print(y)

sns.pairplot(df,hue='diabetic')
plt.show()

N=len(X)
train_size=int(0.8*N)
np.random.seed(42)
indices=np.random.permutation(N)
train_indicies=indices[:train_size]
test_indicies=indices[train_size:]

X_train=X.iloc[train_indicies]
X_test=X.iloc[test_indicies]
y_train=y.iloc[train_indicies]
y_test=y.iloc[test_indicies]

weights=np.random.randn(10,1)
bias=np.random.randn(1,1)
lr=0.1

for epoch in range(1):
    for i in range(len(X_train)):
        x_i=X_train.iloc[i].values.reshape(-1,1)
        y_i=y_train.iloc[i]

        prediction=float(weights.T.dot(x_i)+bias)
        error=y_i-prediction
        
        if prediction*y_i<=0:
            weights=weights+error*lr*x_i
            bias+=error*lr

    if epoch%25==0:
        print(f"loss: {error}")

predictions=[]
for i in range(len(X_test)):
    prediction=float(weights.T.dot(X_test.iloc[i])+bias)
    if prediction>=threshold:
        predictions.append(1)
    else:
        predictions.append(0)

y_true=y_test.values
y_pred=np.array(predictions)
print(y_true)
print(predictions)
TP=np.sum((y_pred==1)&(y_true==1))
TN=np.sum((y_pred==0)&(y_true==0))
FP=np.sum((y_pred==1)&(y_true==0))
FN=np.sum((y_pred==0)&(y_true==1))

conf_matrix=np.array([[TP,FP],[FN,TN]])
print(conf_matrix)

plt.imshow(conf_matrix)
plt.colorbar()

plt.xticks([0,1],["Pred+","Pred-"])
plt.yticks([0,1],["Actual+","Actual-"])

for i in range(2):
    for j in range(2):
        plt.text(j,i,conf_matrix[i,j],
                ha="center", va="center", color="white")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()