# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:13:01 2021

@author: silahi
"""
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as pp

pp.style.use('ggplot')

# Loding iris data 
iris = load_iris()
data = iris['data']
target = iris['target']
labels = iris['target_names']

print("Iris Species :", labels)

"""
 The logistic regression is a binary classifier so we gonna use just 
 two species of iris : the iris-setosa and iris_versicolor witch  correspond
 to the 100 's data points from iris : recall that there is 150 records
 first 50 : setosa
 second 50 : versicolor
 third 50 : virginica
 
 the target varaible contains the classes in this order 
 0 for setosa
 1 for versocolor
 2 for virginica
 
"""

print(data.shape)
dataset = np.hstack((data, target.reshape(-1,1)))
# selection of the 100 s data points (50 for setosa an 50 for versicolor)
dataset = dataset[:100,:]
print(dataset.shape)
print("Dataset : ", dataset, sep=("\n"))

# it's recommanded to shuffle the data for a better learning

for i in range(10):
    np.random.shuffle(dataset)
print(dataset)
# Splitting the dataset in to train set and test set

x_train = dataset[15:,:4]
y_train = dataset[15:, 4]

x_test = dataset[:15,:4]
y_test = dataset[:15, 4]


# Training process
theta = np.random.randn(4)
coefs,costs = gradient_descent(x_train,y_train, theta, etha = 0.01, n_iterations=50)

xc = np.arange(50)
pp.plot(xc, costs)
print("Cost : ", costs[len(costs)-1])

# testin the model 
pred = sigmoid(x_test, coefs)

conf = np.round(pred)== y_test
print("Custom confusion matrix")
print(conf)

# the determination coefficient

def score(x,y,theta):
    u = np.sum(np.square(y - sigmoid(x,theta)))
    v = np.sum(np.square(y - np.mean(y)))
    return 1 - (u/v)

print("Determination coefficient : ", score(x_test,y_test,coefs))
# Defining the logit(linear combination) function 
def logit(x, theta):
    return theta @ x.T

# The sigmoid function
def sigmoid(x,theta):
    u = 1 + np.exp( - logit(x,theta))
    return 1 / u

#The cost function
def cost(x,y,theta) :
    m = len(y)    
    u = y.T @ np.log(sigmoid(x, theta)) 
    v = (1-y).T @ np.log(1-sigmoid(x, theta)) 
    return - (u + v) / m

# Gradient Computation
def gradient(x,y,theta):
    f = sigmoid(x, theta) - y
    return (x.T @ f.reshape(-1,1)).reshape(-1,)

# The gradient descent algorithm
def gradient_descent(x,y,theta,etha =0.001,n_iterations =100) :  
    costs = []
    for i in range(n_iterations) :
        theta = theta - etha * gradient(x, y, theta)
        costs.append(cost(x, y, theta)) 
    return theta,costs    