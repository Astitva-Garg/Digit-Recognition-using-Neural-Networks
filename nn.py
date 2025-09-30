import numpy as np
import pandas as pd
import time

#activation functions

def softmax(x):
    exp = np.exp(x- np.max(x, axis=0, keepdims=True)) # exp can go to inf fast, subrtactic a constant from each term doesent change softmax
    return exp/np.sum(exp, axis = 0, keepdims=True)
    # exp = np.exp(x)
    # return exp/np.sum(exp)

def ReLU(x):
    return np.maximum(0,x)

#algos
def init_params(neurons: int, input_count: int):
    W = np.random.rand(neurons,input_count) - 0.5
    b = np.zeros((neurons, 1)) 
    return W, b

def feedforward(W, X, b, activation_function):
    Z = np.dot(W, X) + b
    return Z, activation_function(Z)

def backprop(dA_l, Y_l, Z_l, W_l, A_l1, d_activation_function): #l represent layer, l1 is l-1, d is derivative
    m = Z_l.shape[1]
    dZ_l = d_activation_function(Z_l,Y_l, dA_l)
    dW_l = np.dot(dZ_l, A_l1.T) / m
    db_l = np.sum(dZ_l, axis = 1, keepdims= True)/m
    dA_l1 = np.dot(W_l.T, dZ_l)
    return dW_l, db_l, dA_l1

def update(W, b, dW, db, alpha):
    W = W - alpha * dW
    b = b - alpha * db
    return W,b

def one_hot(Y): # for each example- assigns the answer to 1 and rest all digits to 0
    arr = np.zeros((Y.size, Y.max() + 1))
    arr[np.arange(Y.size), Y] = 1
    arr = arr.T
    return arr

def gradient_descent(X, Y, iterations, alpha, layers: list):
    input_no = X.shape[0]
    m = X.shape[1]
    
    W1, b1 = init_params(layers[0], input_no)
    W2, b2 = init_params(layers[1], layers[0])

    one_hot_Y = one_hot(Y)
    
    for i in range(iterations):
        Z1, A1 = feedforward(W1, X, b1, ReLU)
        Z2, A2 = feedforward(W2, A1, b2, softmax)
        dW_2, db_2, dA_1 = backprop(None, one_hot_Y, Z2, W2, A1, d_softmax)
        dW_1, db_1, _     = backprop(dA_1, None, Z1, W1, X, d_ReLU)
        W1, b1 = update(W1, b1, dW_1, db_1, alpha)
        W2, b2 = update(W2, b2, dW_2, db_2, alpha)
        if i%10 == 0:
            print(f"iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
            


    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


#derivatives - not actual derivatives, made to work so that we can use combination of activation functions
def d_ReLU(Z, _, dA_L):
    return dA_L*(Z>0)

def d_softmax(Z, Y, _): 
    A = softmax(Z)
    return A - Y

def main():
    toc = time.time()
    train_data = pd.read_csv("Data/train.csv")
    train_data = np.array(train_data)
    np.random.shuffle(train_data)
    train_data = train_data.T
    Y_train = train_data[0]
    X_train = train_data[1:]
    X_train = X_train / 255
    layers =[10,10]
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 5000, 0.5, layers)
    tic = time.time()

    print(tic - toc)

    

main()

