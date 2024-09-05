import tensorflow as tf
from helper import plot_rosenbrock
from keras.optimizers import SGD
from keras.optimizers import Optimizer


class Model:
    def __init__(self) -> None:
        self.x = tf.Variable(
            tf.random.uniform(minval=-2.0, maxval=2.0, shape=[2]),
        )  # x = [x0, x1]
        self.learning_rate = 0.0005
        self.optimizer: Optimizer = SGD(learning_rate=self.learning_rate)
        self.current_loss_val = self.loss()

    def loss(self) -> tf.Tensor:
        self.current_loss_val = (
            100 * (self.x[0] ** 2 - self.x[1]) ** 2 + (self.x[0] - 1) ** 2
        )
        return self.current_loss_val

    def fit(self) -> None:
        with tf.GradientTape() as tape:
            current_loss_val = self.loss()
        gradients = tape.gradient(current_loss_val, self.x)
        self.optimizer.apply_gradients(zip([gradients], [self.x], strict=False))


def main() -> None:
    model = Model()
    gradient_steps = []
    x_start = model.x.numpy()
    num_iterations = 10_000

    for it in range(num_iterations):
        model.fit()
        if it % 500 == 0:
            x = model.x.numpy()
            y = model.current_loss_val.numpy()
            print(f"X = {x} Y = {y}")
            gradient_steps.append(x)

    plot_rosenbrock(x_start, gradient_steps)


if __name__ == "__main__":
    main()
