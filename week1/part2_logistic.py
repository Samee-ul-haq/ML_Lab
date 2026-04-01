#   ----- WEEK 1 part 2 -----
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.datasets import load_diabetes
import math

# Dataframe is a 2D data structure in pandas that can hold data of different types (like integers, floats, strings, etc.) in columns. 
 # It is similar to a table in a relational database or an Excel spreadsheet. Each column in a DataFrame can be thought of as a Series, and the entire DataFrame can be thought of as a collection of Series objects. 
 # The 'columns' parameter is used to specify the names of the columns in the DataFrame, which are taken from the 'feature_names' attribute of the dataset.


dataset=load_diabetes()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['daibetic']=dataset.target

X=df.drop(columns='daibetic') 
y=df['daibetic']


# Why to user a threshold?
# Our perceptron model is made of only one neuron and it adjusts its weights and bias to find better decision boundary.
# It outputs in binary values 0 & 1.SO we are using threshold to convert the continuous output of our model into binary output.


threshold=np.median(y) 
y_binary=np.where(y>threshold,1,0) # converting continuous target variable into binary variable based on the threshold value. 
y=pd.Series(y_binary)
print(y)


# Splitting the dataset into training and testing sets.
# In this way we make sure that our model is evaluated on unseen data, which gives us a better estimate of its performance in real-world scenarios.
train_size=int(0.8*len(X))


# it fixes the randomness of our data splitting process, ensuring that we get the same training and testing sets every time we run the code.
# This is important for reproducibility and debugging purposes.
np.random.seed(42)
indicies=np.random.permutation(len(X))



# Splitting the dataset into training and testing sets using the shuffled indices.
train_indicies=indicies[:train_size]
test_indicies=indicies[train_size:]

X_train=X.iloc[train_indicies]
X_test=X.iloc[test_indicies]
y_train=y.iloc[train_indicies]
y_test=y.iloc[test_indicies]


# defining logistic regression function --- 
# Input parameters:
# X: Input features for training the model.
# y: Target variable for training the model.
# learning_rate: A hyperparameter that controls the step size of parameter updates during gradient descent.
# epochs: The number of times the entire training dataset will be passed through the model during training.
# batch_size: The number of samples processed before the model's parameters are updated during training.
def logisticRegression(X,y,learning_rate,epochs,batch_size): 
    X=np.array(X)
    y=np.array(y).reshape(-1, 1)
    m=len(X)


    # we will also include bias in theta matrix, that will increase the no of rows by 1.
    # new dimension of theta matrix = no of input features + 1 (Another way to include bias nothing more)
    theta=np.random.randn(11,1)

    # we will add a column of 1's to Input features for bias term.
    X_bias=np.c_[np.ones((m,1)),X]



    cost_history=[]  # keep track of cost function.

    # Epochs is the number of times the entire training dataset will be passed through the model during training.
    for epoch in range(epochs):


        # It generates a random permutation of indices from 0 to m-1, which is used to shuffle the data at the beginning of each epoch. 
        # This helps in improving the convergence of the gradient descent algorithm by ensuring that the model does not learn from the data in a fixed order.
        indxs=np.random.permutation(m)          
        X_shuffled=X_bias[indxs]
        y_shuffled=y[indxs]


        #Inside the each epoch,we iterate through the entire shuffled training dataset.
        # We process the data in batches of size 'batch_size' to update the model parameters (theta) using gradient descent.
        # which means after every batch we are updating the theta values towards the direction of minimum cost function.
        for i in range(0,m,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            y_batch=y_shuffled[i:i+batch_size]


            # using result derived of gradient descent.
            gradients=2/batch_size*\
                X_batch.T.dot(X_batch.dot(theta)-y_batch)
            

            # updating theta values towards the direction of minimun cost function.
            # learning_rate is a hyperparameter (usually a small positive value) that controls the step size of the parameter updates during gradient descent.
            # If the learning rate is too large, the model may overshoot the optimal parameters and diverge, while if it is too small, the model may take a long time to converge.
            # ANALOGY : Take the analogy of a monster rolling down a hill to find the lowest point.
            #           IF the monster takes very large steps,he may jump from one hill to another and never reach the lowest point.
            #           On the other hand, if he takes very small steps,he will take a long time to reach the lowest point.
            theta=theta-learning_rate*gradients


        # calculatiog predictions after updating all the theta values -- for one epoch.
        predictions=X_bias.dot(theta)
        

        # calculatiing cost function using log likelihood function for logistic regression.
        cost=-(y.T.dot(np.log(predictions))+(1-y).T.dot(np.log(1-predictions)))
        cost_history.append(cost)
        

        # printing cost value after every 100 epochs to see how our model is learning.
        if epoch%100==0:
            print(f"Epoch {epoch},Cost:{cost}")

    return theta,cost_history

theta_final,cost_history=logisticRegression(X_train,y_train,learning_rate=0.001,epochs=1000,batch_size=64)


plt.plot(np.array(cost_history).flatten())
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