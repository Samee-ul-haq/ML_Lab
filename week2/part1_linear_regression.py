from sklearn.datasets import fetch_california_housing # skit-learn constains numerous datasets for testing and learning purposes.
import numpy as np # Numerical computing.
import matplotlib # Creating Data visualizations.
matplotlib.use('TkAgg') # Use TkAgg backend for matplotlib to enable plotting in environments without a display server.
import matplotlib.pyplot as plt # Creating Data visualizations.
import pandas as pd # Data handling & manipulation.


# dataset is loaded using inbuild fetch_california_housing() function from sklearn.datasets.
# This dataset contains information about California housing prices and is commonly used for regression tasks in machine learning.
dataset = fetch_california_housing()



# Dataframe is a 2D data structure in pandas that can hold data of different types (like integers, floats, strings, etc.) in columns. 
# It is similar to a table in a relational database or an Excel spreadsheet. 
# Each column in a DataFrame can be thought of as a Series, and the entire DataFrame can be thought of as a collection of Series objects. 
# The 'columns' parameter is used to specify the names of the columns in the DataFrame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)




# Function to normalize the data using mean and standard deviation.
# -----------   Why Normalization?   ----------------
# Take Example :-----  Number of Bedrooms: Usually a small number (1, 2, 3, 4).Square Footage: A much larger number (800, 1500, 3000,2000).
# Because 3000 is mathematically much larger than 3, the model might mistakenly think the square footage is 1000 times more important than the number of bedrooms.
#  ----------   How Normalization Fixes This ------
# To normalize, we squeeze both sets of numbers into small numbers usually between and around -1 and 1.
# Take the above two examples as input features and we will apply normalization to them.
# mean_bedrooms = 2.5, std_bedrooms = 1.0, mean_square_footage = 1824, std_square_footage = 1000.
# Normalized Bedroom  = (3 - 2.5) / 1.0 = 0.5, Normalized Square Footage = (3000 - 1500) / 1000 = 1.5.
# Feature          Original                ValuesStandardized (normalized)
# Bedrooms        [1, 2, 3, 4]               [-1.34, -0.45, 0.45, 1.34]
# Sq Footage      [800, 1500, 3000, 2000]    [-1.28, -0.41, 1.47, 0.22]
def normalize(X):
    X_arr = np.array(X)
    if len(X_arr.shape) == 1:
        X_arr = X_arr.reshape(-1, 1)
    mean = np.mean(X_arr, axis=0)
    std = np.std(X_arr, axis=0)
    std_replaced = np.where(std == 0, 1, std)
    return (X_arr - mean) / std_replaced, mean, std

X_all, mean_all, std_all = normalize(df.values)
X_one, mean_one, std_one = normalize(df[['MedInc']].values)
y = dataset.target.reshape(-1, 1)



# Class definition for linear regression model.
class linearRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iters = iterations
        self.weights = None
        self.bias = 0
        self.loss_history = []
        self.theta_history = []
    
    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.loss_history = []
        self.theta_history = []

        for i in range(self.iters):
            Y_pred = np.dot(X, self.weights) + self.bias
            
            error = Y_pred - y
            loss = (1 / (2 * n_samples)) * np.sum(error**2)
            self.loss_history.append(loss)
            
            if n_features == 1:
                self.theta_history.append((self.bias, self.weights[0, 0]))

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias

m1 = linearRegression(learning_rate=0.01, iterations=1000)
m1.fit(X_one, y)

m2 = linearRegression(learning_rate=0.01, iterations=1000)
m2.fit(X_all, y)

plt.figure(figsize=(10, 5))
plt.plot(m1.loss_history, label=f'Model 1 (1 Feature) Final: {m1.loss_history[-1]:.4f}')
plt.plot(m2.loss_history, label=f'Model 2 (8 Features) Final: {m2.loss_history[-1]:.4f}')
plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig('loss_comparison.png')

plt.figure(figsize=(8, 5))
plt.scatter(X_one[:200], y[:200], alpha=0.5, label='Actual Data')
plt.plot(X_one[:200], m1.predict(X_one[:200]), color='red', linewidth=3, label='Regression Line')
plt.title("Model 1: MedInc vs Price")
plt.xlabel("Normalized Median Income")
plt.ylabel("House Price")
plt.legend()
plt.savefig('plot2.png')

b_range = np.linspace(m1.bias - 2, m1.bias + 2, 50)
w_range = np.linspace(m1.weights[0,0] - 2, m1.weights[0,0] + 2, 50)
B, W = np.meshgrid(b_range, w_range)
Z = np.zeros(B.shape)
for i in range(len(b_range)):
    for j in range(len(w_range)):
        y_p = X_one * w_range[j] + b_range[i]
        Z[j, i] = (1 / (2 * len(y))) * np.sum((y_p - y)**2)

plt.figure(figsize=(8, 6))
plt.contour(B, W, Z, levels=30)
path = np.array(m1.theta_history)
plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=3, label='GD Path')
plt.title("Model 1: Loss Contour and GD Path")
plt.xlabel("Bias (θ0)")
plt.ylabel("Weight (θ1)")
plt.legend()
plt.savefig('plot3.png')

rates = [0.001, 0.005, 0.01, 0.05]
plt.figure(figsize=(10, 5))
for r in rates:
    temp_model = linearRegression(learning_rate=r, iterations=100)
    temp_model.fit(X_one, y)
    plt.plot(temp_model.loss_history, label=f'η = {r}')
plt.title("Effect of Learning Rates on Convergence")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('plot 4.png')