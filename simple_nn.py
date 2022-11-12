import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import pickle
import csv



def load_data(filename):
    train = []
    labels = []
    file = open(filename, "r")
    reader = csv.reader(file)
    for row in reader:
        label = int(row[-1])
        row = [float(i) for i in row[:-1]]
        row = np.array(row).reshape(784, 1)
        train.append(row)
        labels.append(label)
    file.close()
    return train, labels


class Cost:
    def __init__(self, cost):
        self.cost = cost
    
    def cost_derivative(self, a, y, z = None):
        if self.cost == "quadractic":
            return self.quadractic_cost(a, y, z)
        return self.cross_entropy(a, y)

    def quadractic_cost(self, a, y, z):
        return (a - y) * sigmoid_prime(z)
    
    def cross_entropy(self, a, y):
        return a - y

class Layer:
    def __init__(self, weights, biases, act = "sigmoid"):
        self.init_act(act)
        self.weights = weights
        self.biases = biases

    def init_act(self, act):
        if act == "sigmoid":
            self.act = self.sigmoid
            self.act_prime = self.sigmoid_prime
        elif act == "relu":
            self.act = self.relu
            self.act_prime = self.relu_prime
        elif act == "softmax":
            self.act = self.softmax
            self.act_prime = self.softmax_prime
    
    
    def sigmoid(self, z):
        """The sigmoid function.""" 
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function.""" 
        return (1 - self.sigmoid(z))  * self.sigmoid(z)
    
    def relu_prime(self, z):
         z[z<=0] = 0
         z[z>0] = 1
         return z

    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        e = np.exp(z - np.max(z))
        return e/e.sum()
    
    def softmax_prime(self, z):
        return self.softmax(z) * (1 - self.softmax(z))


# a function to change the a list of label values [1,2,3,4] to a list of 
# [0, 1 ,...] 1 mean the index of the label value
# one hot encoding
def vectorized_result(label):
    e = np.zeros((10 ,1))
    e[label] = 1.0
    return e

def sigmoid(z):
        """The sigmoid function.""" 
        return 1.0/(1.0+np.exp(-z))
        
def sigmoid_prime(z):
        """Derivative of the sigmoid function.""" 
        return sigmoid(z)*(1-sigmoid(z))
        
def relu_prime(z):
         z[z<=0] = 0
         z[z>0] = 1
         return z

def relu(z):
    return np.maximum(0, z)





class NN:
    def __init__(self, sizes, lmbda = 0.1, acts = [], cost = Cost("quadractic")):
        self.num_of_layers = len(sizes)
        self.layers = []
        self.cost = cost
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.m = 0.8
        self.lmbda = lmbda
        self.decay = 0.0016
        self.init_layers(acts)
    
    def init_layers(self, acts):
        act = "sigmoid"
        for i in range(0, self.num_of_layers - 1):
            if len(acts) >= i + 1:
                act = acts[i]
            self.layers.append(Layer(self.weights[i], self.biases[i], act))

                
            

    def train(self, train_data, epoch, batch_size,eta, eva = None):
        for e in range(epoch):
            random.shuffle(train_data)
            mini_batches = []
            #eta = eta * 1/(1+ self.decay * epoch)
            for i in range(0, len(train_data), batch_size):
                mini_batches.append(train_data[i : i +  batch_size])
            for batch in mini_batches:
                self.update(batch, eta, 0.5, len(train_data))
            print(f"Epoch {e} Training size : {len(train_data)}\nTraining Accuracy : {self.accuracy(train_data) / len(train_data) * 100:.2f}%")
            if eva:
                n_test = len(eva)
                print("Test Accuracy : {0:.2f}% / Size : {1}".format( self.evaluate(eva) / n_test * 100, n_test))
            print("===============================================================")
    
    def feedforward(self, a):
        for w, b, l in zip(self.weights, self.biases, self.layers):
            a = l.act((np.dot(w, a) + b))
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def backprop(self, var, label):
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]
        activation = var
        activations = [activation]
        zs = []
        for w, b, l in zip(self.weights, self.biases, self.layers):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = l.act(z)
            activations.append(activation)
        #delta = self.cost_derivative2(activations[-1], label)
        delta = self.cost.cost_derivative(activations[-1], label, zs[-1])
        n_w[-1] = np.dot(delta, activations[-2].transpose())
        for last in range(2, self.num_of_layers):
            delta = np.dot(self.weights[-last + 1].transpose(), delta) * self.layers[-last].act_prime(zs[-last]) 
            n_w[-last] = np.dot(delta, activations[-last-1].transpose())
            n_b[-last] = delta
        return (n_w, n_b)

    def update(self, train_data, eta, lmbda, n):
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]
        for var, lab in train_data:
            nd_w, nd_b = self.backprop(var, lab)
            n_w, n_b = nd_w, nd_b
        #self.weights  = [     
        #        old_w - (eta / len(train_data) * new_w) for old_w, new_w in zip(self.weights, n_w)
        #    ]
        self.weights = [
            (1-eta*(lmbda/n))*old_w-(eta/len(train_data))*new_w for old_w, new_w in zip(self.weights, n_w)
        ]
        self.biases = [
                old_b - (eta / len(train_data) * new_b) for old_b, new_b in zip(self.biases, n_b)
            ]

    def cost_derivative(self, output_activations, y):
        # the change in Cost with respect to activation
        return output_activations - y
    

    def cost_derivative2(self, yhat, y):
        return yhat - y

    def cost_derivative3(self, output_activations, y, z):
        # the change in Cost with respect to activation
        return output_activations - y * self.layers[-1].act_prime(z)
    
    def accuracy(self, train_data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x,y in train_data]
        return sum(int(x == y)for x, y in results)

#
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
##x_train, x_test = x_train / 255.0, x_test/255.0
#x_train = [
#    x.reshape(784, 1) for x in x_train
#]
#y_train = [
#    vectorized_result(y) for y in y_train
#]
##
#x_test = [
#    x.reshape(784, 1) for x in x_test
#]
#
#dataset = [
#    (x, y) for x, y in zip(x_train, y_train)
#]
#
#testset = [
#    (x, y) for x, y in zip(x_test, y_test)
#]
##net = NN([784, 100, 75, 50, 25, 10], ["relu", "sigmoid", "sigmoid", "sigmoid"])
#net = NN([784, 100, 75, 50, 25, 10], [])
##net.train(dataset, 20, 10, 0.23, testset)
#net.train(dataset, 20, 10, 0.23, testset)
#file = open("nn.pkl", "wb+")
#pickle.dump(net, file)
#file.close()