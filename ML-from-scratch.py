#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:02:24 2023

@author: 33574
"""

import numpy as np
from tqdm import tqdm


class LinearRegression():

    def __init__(self,lr,n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit_model(self,X,y):
        ## Use Gradient Descent to train
        n_samples,n_features = X.shape
        #print(n_samples, n_features)
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in tqdm(range(self.n_iter)):

            # Calculate prediction
            y_pred = np.dot(x,self.weights) + self.bias

            # Compute Gradients
            delw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            delb = (1/n_samples)*np.sum(y_pred-y) 

            # Update weights and biases
            self.weights = self.weights - self.lr*delw
            self.bias = self.bias - self.lr*delb
        
        return

    def predict(self,X):
        return (np.dot(X,self.weights) + self.bias)



class LogisticRegression():
    
    def __init__(self,lr,n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    def fit_model(self,X,y):
        n_samples,n_features = X.shape
        
        # weights and bias initialization
        
        self.weights = np.random.rand(n_features)
        self.bias = 0
        
        # Start training
        
        for _ in range(self.n_iter):
            linear_output = np.dot(X,self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)
            
            # Gradient computation
            delw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            delb = (1/n_samples)*np.sum(y_pred-y)
            
            # Update weights
            
            self.weights = self.weights - self.lr*delw
            self.bias = self.bias - self.lr*delb
    
    def predict_class(self,X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_output)
        y_pred_class = [1 if i>0.5 else 0 for i in y_pred]
        return y_pred_class