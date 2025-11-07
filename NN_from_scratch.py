import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# 0. NN parameters
max_epochs = 200

dim_input = 4
dim_hidden1 = 8
dim_hidden2 = 8
dim_output = 3  # Setosa, Versicolor, Virginica

learning_rate = 0.1


# 1. dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target

x = iris_df.iloc[:, :4].values # feature
y = iris_df['class'].values    # target


# 2. Normalization, Encoding, train/test split
def normalize(x):
    std = x.std(axis=0)
    mean = x.mean(axis=0)
    return (x-mean)/std

def one_hot_encoding(y):
    n_classes = np.max(y)+1
    return np.eye(n_classes)[y]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = normalize(x_train)
x_test = normalize(x_test)
y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)


# 3. Layers
class Layer:
    
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2.0 / n_input)
        self.bias   = np.zeros((1, n_neurons))

    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output

hidden_layer_1 = Layer(dim_input, dim_hidden1)
hidden_layer_2 = Layer(dim_hidden1, dim_hidden2)
output_layer   = Layer(dim_hidden2, dim_output)


# 4. Activation Functions
def ReLu(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z = np.exp(z-np.max(z,   axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def ReLU_derivative(z):
    return (z > 0).astype(float) 


# 5. Loss Function (CCE)
def cross_entropy(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# 6. Train
def train(epochs, learning_rate,
          x_train, y_train,
          hidden_layer_1, hidden_layer_2, output_layer):
    
    losses = []
    for epoch in range(epochs):

        z1 = hidden_layer_1.forward_propagation(x_train)
        A1 = ReLu(z1)
        z2 = hidden_layer_2.forward_propagation(A1)
        A2 = ReLu(z2)
        z_hat = output_layer.forward_propagation(A2)
        y_pred = softmax(z_hat)

        J = cross_entropy(y_pred, y_train)
        losses.append(J)

        # backward propagation
        n_obs = x_train.shape[0]

        # gradients output
        dJ_dzhat = (y_pred - y_train) / n_obs
        dJ_w3 = np.dot(A2.T, dJ_dzhat) 
        dJ_db3 = np.sum(dJ_dzhat, axis=0, keepdims=True) 
        dJ_dA2 = np.dot(dJ_dzhat, output_layer.weights.T)

        # gradients hidden 2
        dJ_dz2 = dJ_dA2 * ReLU_derivative(z2)
        dJ_dw2 = np.dot(A1.T, dJ_dz2) 
        dJ_db2 = np.sum(dJ_dz2, axis=0, keepdims=True) 
        dJ_dA1 = np.dot(dJ_dz2, hidden_layer_2.weights.T)

        # gradients hidden 1
        dJ_dz1 = dJ_dA1 * ReLU_derivative(z1)
        dJ_dw1 = np.dot(x_train.T, dJ_dz1)
        dJ_db1 = np.sum(dJ_dz1, axis=0, keepdims=True) 

        # update W and b
        output_layer.weights -= learning_rate * dJ_w3 
        hidden_layer_2.weights -= learning_rate * dJ_dw2
        hidden_layer_1.weights -= learning_rate * dJ_dw1

        output_layer.bias -= learning_rate * dJ_db3 
        hidden_layer_2.bias -= learning_rate * dJ_db2
        hidden_layer_1.bias -= learning_rate * dJ_db1

    return losses

losses = train(max_epochs, learning_rate, x_train, y_train, 
      hidden_layer_1, hidden_layer_2, output_layer)


# 7. Predict
def predict(x, hidden_layer_1, hidden_layer_2, output_layer):
    z1 = hidden_layer_1.forward_propagation(x)
    A1 = ReLu(z1)
    z2 = hidden_layer_2.forward_propagation(A1)
    A2 = ReLu(z2)
    z_hat = output_layer.forward_propagation(A2)
    y_pred = softmax(z_hat)
    
    return y_pred

y_pred_train = predict(x_train, hidden_layer_1, hidden_layer_2, output_layer)
y_pred_test  = predict(x_test,  hidden_layer_1, hidden_layer_2, output_layer)


# 8. Resluts
def accuracy(y_prob, y_true):
    y_pred_cls = np.argmax(y_prob, axis=1)
    y_true_cls = np.argmax(y_true, axis=1)
    return np.mean(y_pred_cls == y_true_cls)

print(f"Final train loss: {losses[-1]:.4f}")
print(f"Train accuracy:   {accuracy(y_pred_train, y_train):.3f}")
print(f"Test  accuracy:   {accuracy(y_pred_test,  y_test):.3f}")

y_test_cls = np.argmax(y_test, axis=1)
y_hat_cls  = np.argmax(y_pred_test, axis=1)
print('Confusion matrix (test):')
print(confusion_matrix(y_test_cls, y_hat_cls))
