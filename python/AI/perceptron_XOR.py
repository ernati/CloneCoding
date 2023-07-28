import random
import matplotlib.pyplot as plt
from math import exp,log

import numpy as np

# 입력값 및 라벨값 선언 및 초기화
X = [ (0,0), (1,0), (0,1), (1,1) ]
Y_AND = [ 0,0,0,1 ]
Y_OR = [ 0,1,1,1 ]
Y_XOR = [ 0,1,1,0 ]

class logistic_regression_model() :
    def __init__(self) :
        self.w = np.random.random( (2,1) )
        self.b = random.random()

    def sigmoid(self,z) :
        return 1/( 1+exp(-z) )
    
    def predict(self,x) :
        z = self.w[0]*x[0] + self.w[1]*x[1] + self.b
        a = self.sigmoid(z)
        return a
    
model = logistic_regression_model()

def train(X,Y,model,lr):
    dw0 = 0.0
    dw1 = 0.0
    db = 0.0
    m = len(X)
    cost = 0.0
    for x,y in zip(X,Y) :
        a = model.predict(x)
        if y==1 :
            cost -= log(a)
        else :
            cost -= log(1-a)
        
        dw0 += (a-y)*x[0]
        dw1 += (a-y)*x[1]
        db += (a-y)
    
    cost /= m
    model.w[0] -= lr * dw0/m
    model.w[1] -= lr * dw1/m
    model.b -= lr*db/m

    return cost


model.predict( (1,1) )
