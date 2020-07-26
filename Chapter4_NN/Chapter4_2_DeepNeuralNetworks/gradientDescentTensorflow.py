import tensorflow as tf

from helper import plot_rosenbrock


class Model:
    def __init__(self):
        self.x = tf.Variable(tf.random.uniform(shape=[2], minval=-2.0, maxval=2.0)) # x = [x0, x1]
        self.learning_rate = 0.001 # eta
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate) # SGD = (stochastic) gradient descent
        self.current_loss_val = self.loss()
    
    def loss(self):
        self.current_loss_val = 100 * (self.x[0]**2 - self.x[1])**2 + (self.x[0] - 1)**2
        return self.current_loss_val
    
    def fit(self):
        self.optimizer.minimize(self.loss, self.x) # loss function, variables 
    
model = Model()
gradient_steps = []

for it in range(5000):
    model.fit()
    if it % 100 == 0:
        print(model.x.numpy(), model.current_loss_val.numpy())
        gradient_steps.append(model.x.numpy())

plot_rosenbrock(x_start=gradient_steps[0], gradient_steps=gradient_steps)
