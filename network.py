import math
import random

def sigmoid(x):
    value = 1 / (1 + math.exp(-x))
    return value

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def d_relu(x):
    if x > 0:
        return 1
    else:
        return 0
class TinyNN:

    def __init__(self, lr=0.1):

        self.lr = lr
        self.W1 = []

        i = 0
        while i < 3:
            row = []
            j = 0
            while j < 2:
                row.append(random.uniform(-0.1, 0.1))
                j = j + 1
            self.W1.append(row)
            i = i + 1

        self.b1 = [0, 0, 0]

        self.W2 = []
        i = 0
        while i < 3:
            self.W2.append(random.uniform(-0.1, 0.1))
            i = i + 1

        self.b2 = 0

    def forward(self, x):

        self.z1 = []
        self.a1 = []
        i = 0
        while i < 3:
            z = 0
            z = z + (self.W1[i][0] * x[0])
            z = z + (self.W1[i][1] * x[1])
            z = z + self.b1[i]
            self.z1.append(z)
            activated = relu(z)
            self.a1.append(activated)
            i = i + 1

        z2 = 0
        i = 0
        while i < 3:
            z2 = z2 + (self.W2[i] * self.a1[i])
            i = i + 1
        z2 = z2 + self.b2
        self.z2 = z2
        self.y_hat = sigmoid(z2)

        return self.y_hat

    def backward(self, x, y):
        
        error = self.y_hat - y
        dW2 = []
        i = 0
        while i < 3:
            dW2.append(error * self.a1[i])
            i = i + 1
        db2 = error

        d_hidden = []
        i = 0
        while i < 3:
            temp = error * self.W2[i]
            temp = temp * d_relu(self.z1[i])
            d_hidden.append(temp)
            i = i + 1

        dW1 = []
        i = 0
        while i < 3:
            row = []
            row.append(d_hidden[i] * x[0])
            row.append(d_hidden[i] * x[1])
            dW1.append(row)
            i = i + 1

        db1 = d_hidden
        i = 0
        while i < 3:
            self.W2[i] = self.W2[i] - (self.lr * dW2[i])
            i = i + 1
        self.b2 = self.b2 - (self.lr * db2)
        i = 0
        while i < 3:
            j = 0
            while j < 2:
                self.W1[i][j] = self.W1[i][j] - (self.lr * dW1[i][j])
                j = j + 1

            self.b1[i] = self.b1[i] - (self.lr * db1[i])
            i = i + 1

    def save(self, filename):
        file = open(filename, "w")
        file.write("W1=" + str(self.W1) + "\n")
        file.write("b1=" + str(self.b1) + "\n")
        file.write("W2=" + str(self.W2) + "\n")
        file.write("b2=" + str(self.b2) + "\n")
        file.close()

    def load(self, filename):
        file = open(filename)
        for line in file:
            parts = line.split("=")
            name = parts[0]
            value = parts[1]
            setattr(self, name, eval(value))
            
        file.close()