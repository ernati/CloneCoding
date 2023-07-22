import torch
import numpy as np
import matplotlib.pyplot as plt
import math

#x_seeds : 저 값을 가지는 numpy배열 ( 그냥 벡터..랄까? )
x_seeds = np.array( [ (0,0),(1,0),(0,1),(1,1) ], dtype = np.float )
y_seeds = np.array( [0,1,1,0] )

N = 1000
#idxs : 0이상 4미만의 범위에서 N개의 임의의 정수 array
idxs = np.random.randint(0,4,N)

#X와 y는 x_seeds의 요소들로 이루어진 array인데, 그 내용물의 x_seeds의 index는 idxs다.
X = x_seeds[idxs]
Y = y_seeds[idxs]

# 표준편차가 0.25인 정규분포에서 X.shape만큼의 array를 만들어서 X에 더해준다.
X += np.random.normal( scale = 0.25, size = X.shape )

class shallow_neural_network():
    def __init__(self, num_input_features, num_hiddens):
        self.num_input_features = num_input_features
        self.num_hiddens = num_hiddens

        self.W1 = np.random.normal( size = (num_hiddens, num_input_features) )
        self.b1 = np.random.normal( size = num_hiddens )
        self.W2 = np.random.normal( size = num_hiddens )
        self.b2 = np.random.normal( size = 1 )

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    
    def predict(self,x):
        #matmul : 행렬곱
        z1 = np.matmul(self.W1,x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.matmul(self.W2, a1) + self.b2
        a2 = self.sigmoid(z2)
        return a2, (z1,a1,z2,a2)
    
def train(X,Y,model,lr=0.1) :
    dW1 = np.zeros_like(model.W1)
    db1 = np.zeros_like(model.b1)
    dW2 = np.zeros_like(model.W2)
    db2 = np.zeros_like(model.b2)
    m = len(X)
    cost = 0.0

    for x,y in zip(X,Y) :
        a2, (z1, a1, z2, _) = model.predict(x)
        if y==1 :
            cost -= np.log(a2)
        else :
            cost -= np.log(1-a2)
    
        db2 = db2 + (a2-y)
        dW2 = dW2 + (a2-y) * a1

        e1 = 1 - (a1*a1)

        db1 = db1 + (a2-y) * model.W2 + e1
        #np.outer : 외적
        dW1 = dW1 + np.outer( (a2-y) * model.W2 * e1, x )
    
    cost /= m
    model.W1 -= lr * dW1 / m
    model.b1 -= lr * db1 / m
    model.W2 -= lr * dW2 / m
    model.b2 -= lr * db2 / m

    return cost

model = shallow_neural_network(2,3)

for epoch in range(100) :
    cost = train(X,Y,model, 0.1)
    if epoch % 10 == 0 :
        print(epoch, cost)

model.predict( (1,1) )[0].item()
model.predict( (1,0) )[0].item()
model.predict( (0,1) )[0].item()
model.predict( (0,0) )[0].item()



