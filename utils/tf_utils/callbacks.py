import numpy as np
import tensorflow as tf


def schedule_fn(epoch: int) -> float:
    learning_rate = 1e-3
    if epoch < 5:
        learning_rate = 1e-3
    elif epoch < 20:
        learning_rate = 5e-4
    else:
        learning_rate = 1e-4
    return learning_rate


def schedule_fn2(epoch: int) -> float:
    if epoch < 10:
        return 1e-3
    else:
        return 1e-3 * np.exp(0.1 * (10 - epoch))


def schedule_fn3(epoch: int) -> float:
    return 1e-3 * np.exp(0.1 * (10 - epoch))


def schedule_fn4(epoch: int) -> float:
    return 1e-3 * np.exp(0.05 * (10 - epoch))


def schedule_fn5(epoch: int) -> float:
    learning_rate = 1e-3
    if epoch < 100:
        learning_rate = 1e-3
    elif epoch < 200:
        learning_rate = 5e-4
    elif epoch < 300:
        learning_rate = 1e-4
    else:
        learning_rate = 5e-5
    return learning_rate


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir: str, **kwargs: dict) -> None:
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        logs.update({"learning_rate": self.model.optimizer.learning_rate.numpy()})
        super().on_epoch_end(epoch, logs)
