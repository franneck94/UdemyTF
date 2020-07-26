import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

# Dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
y_train = y_train.reshape(-1, 1)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
y_test = y_test.reshape(-1, 1)

# Dataset variables
features = x_train.shape[1]
target = y_train.shape[1]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# Model variables
hidden_layer_size = 50
nodes = [features, hidden_layer_size, target] # input, hidden, output
epochs = 2200

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped

class Model:
    def __init__(self):
        # Weights (Matrices)
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=[nodes[0], nodes[1]], stddev=0.1), name="W1")
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=[nodes[1], nodes[2]], stddev=0.1), name="W2")
        # Biases (Vectors)
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]), name="b1")
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]), name="b2")
        self.variables = [self.W1, self.W2, self.b1, self.b2]
        # Model variables
        self.learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.current_loss_val = None
        
    def get_variables(self):
        return {var.name: var.numpy() for var in self.variables}
        
    def predict(self, x):
        input_layer = x
        hidden_layer = tf.math.add(tf.linalg.matmul(input_layer, self.W1), self.b1)
        hidden_layer_act = tf.nn.relu(hidden_layer)
        output_layer = tf.math.add(tf.linalg.matmul(hidden_layer_act, self.W2), self.b2)
        return output_layer
    
    def loss(self, y_true, y_pred):
        loss_fn = tf.math.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred))
        self.current_loss_val = loss_fn.numpy()
        return loss_fn
    
    def update_variables(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss = self.loss(y_train, y_pred)
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables)) 
        return loss
    
    def compute_metrics(self, x, y):
        y_pred = self.predict(x)
        return r_squared(y, y_pred)  
    
    def fit(self, x_train, y_train, epochs=10):
        #print("Weights at the start: ", self.get_variables())
        for epoch in range(epochs):
            train_loss = self.update_variables(x_train, y_train).numpy()
            train_r2 = self.compute_metrics(x_train, y_train).numpy()
            if epoch % 100 == 0:
                print("Epoch: ", epoch+1, " of ", epochs,
                      " - Train Loss: ", round(train_loss, 4),
                      " - Train R2: ", round(train_r2, 4))
        #print("Weights at the end: ", self.get_variables())
        
    def evaluate(self, x, y):
        loss = self.loss(self.predict(x), y).numpy()
        r2 = self.compute_metrics(x, y).numpy()
        print("Loss: ", round(loss, 4), " R2: ", round(r2, 4))
             
model = Model()
model.fit(x_train, y_train, epochs=epochs)
model.evaluate(x_test, y_test)