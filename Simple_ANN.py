# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:11:52 2019

@author: arpit.agrawal
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

X = np.array([[0,1],[1,0],[0,0],[1,1]])
y = np.array([[1],[1],[0],[0]])

input_node = 2
hidden_node = 5
output_node = 1


wxh = np.random.rand(input_node, hidden_node)
bh = np.zeros((1,hidden_node))
why = np.random.rand(hidden_node,output_node)
bo  = np.zeros((1,output_node))

def sigmoid(x):
    return( 1/(1+np.exp(-x)))
    
def delta_sigmoid(x):
    return((np.exp(-x))/(1+np.exp(-x))**2)

def forward_pass(X,wxh,why):
    z1 = np.dot(X,wxh) + bh
    a1 = sigmoid(z1)
    z2 = np.dot(a1,why) + bo
    yhat = sigmoid(z2)
    return(z1,a1,z2,yhat)
    
def cost_function(y,yhat):
    return(0.5*(sum(y-yhat)**2))

def backward_pass(yhat,z2,a1,z1):
    delta_2 = np.multiply(-(y-yhat),delta_sigmoid(z2))
    dJ_dwhy = np.dot(a1.T,delta_2)
    delta_1 = np.dot(delta_2,why.T)*delta_sigmoid(z1)
    dJ_dwxh = np.dot(X.T, delta_1)
    return(dJ_dwxh,dJ_dwhy)
    
alpha = 0.001
num_iterations = 20000

cost =[]
for i in range(num_iterations):
    z1,a1,z2,yhat = forward_pass(X,wxh,why)
    dJ_dwxh,dJ_dwhy = backward_pass(yhat,z2,a1,z1)

    wxh = wxh - alpha * dJ_dwxh
    why = why - alpha * dJ_dwhy
    
    k = cost_function(y,yhat)
    cost.append(k)
    print(k)
    
plt.grid()
plt.plot(range(num_iterations), cost)
    
plt.title('Cost Function')
plt.xlabel('Training Iterations')
plt.ylabel('Cost')    
    
    