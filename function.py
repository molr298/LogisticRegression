import pandas as pd
import numpy as np
import map_feature as mf
import json

class LogisticRegression:
    def __init__(self, file_input):
        self.df = pd.read_csv(file_input, names=["col1", "col2", "label"])
        self.X = mf.map_feature(self.df.col1.values, self.df.col2.values)
        self.Y = self.df.label.values
        f = open('config.json')
        self.config = json.load(f)
        f.close()
        print(self.config )
        self.Lambda = self.config['Lambda']
        self.Alpha = self.config['Alpha']
        self.NumIter = self.config['NumIter']
        self.m = self.X.shape[0]
        self.n =self.X.shape[1]

    def compute_cost(self, theta):
        h_theta = self.h_theta(np.dot(self.X, theta))
        J = (-self.Y)*np.log(h_theta) + (1 - self.Y)*np.log(1 - h_theta) + (self.Lambda/2*self.m)*np.sum(np.pow(theta,2))
        return J.mean()

    def compute_gradient(self, j, theta):
        h_x = self.h_theta(np.dot(self.X, theta))
        gradient_vector = 0.0
        for i in range(h_x.shape[0]):
            gradient_vector = gradient_vector + (h_x[i] - self.Y[i]) * self.X[i][j]
        return gradient_vector
    
    def h_theta(self, x):
        return 1/(1 + np.exp(-x))

    def gradient_descent(self):
        theta = np.zeros(self.n)
        for _ in range(self.NumIter):
            for j in range(self.n):
                gradient = self.compute_gradient(j, theta)
                pre_theta = theta[j]
                if j >0:
                    gradient += (1/self.m)*pre_theta
                theta[j] = pre_theta - self.Alpha*gradient
        return theta

    def fit(self):
        self.theta = self.gradient_descent()

    def predict(self):
        pass
    def evaluate(self):
        pass