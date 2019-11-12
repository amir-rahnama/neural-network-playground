import numpy as np

def loss(y_hat, y):
    return np.mean(np.abs(y_hat - y))

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def d_sigmoid(output):
    return output*(1-output)
    

def forward_propagation(a_0, w_0, w_1):
    z_1 = np.matmul(a_0, w_0)
    a_1 = sigmoid(z_1)

    z_2 = np.matmul(a_1,w_1)
    a_2 = sigmoid(z_2)

    return a_1, a_2

def back_propagation(w_1, a_1, a_2, y):
    a_2_error = a_2 - y
    layer_2_delta = np.multiply(a_2_error, d_sigmoid(a_2))

    layer_1_error = np.matmul(layer_2_delta, w_1.T)
    layer_1_delta = np.multiply(layer_1_error, d_sigmoid(a_1))

    return layer_1_delta, layer_2_delta


def train(x, y, hidden_size, alpha=1, num_iter = 60000):
    w_0 = 2 * np.random.random((2, hidden_size)) - 1
    w_1 = 2 * np.random.random((hidden_size, 1)) - 1
    
    loss_values = []

    for i in range(num_iter):
        a_0 = x
        a_1, a_2 = forward_propagation(a_0, w_0, w_1)
    
        layer_1_delta, layer_2_delta = back_propagation(w_1, a_1, a_2, y)

        w_1 -= alpha * np.matmul(a_1.T, layer_2_delta)
        w_0 -= alpha * np.matmul(a_0.T, layer_1_delta)

        if i % 1000 == 0:
            loss_values.append(loss(a_2, y))
        
    return {'loss': loss_values, 'w_0': w_0, 'w_1': w_1}
