import io
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def display_digit(
    image: np.ndarray,
    label: np.ndarray = None,
    pred_label: np.ndarray = None
) -> None:
    """Display the MNIST image.
    If the *label* and *label* is given, these are also displayed.

    Parameters
    ----------
    image : np.ndarray
        MNIST image.
    label : np.ndarray, optional
        One-hot encoded true label, by default None
    pred_label : np.ndarray, optional
        One-hot encoded prediction, by default None
    """
    if image.shape == (784,):
        image = image.reshape((28, 28))
    label = np.argmax(label, axis=0)
    if pred_label is None and label is not None:
        plt.figure_title(f'Label: {label}')
    elif label is not None:
        plt.figure_title(f'Label: {label}, Pred: {pred_label}')
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_digit_and_predictions(
    image: np.ndarray,
    label: int,
    pred: int,
    pred_one_hot: np.ndarray
) -> None:
    """Display the MNIST image and the predicted class as a title.

    Parameters
    ----------
    image : np.ndarray
        MNIST image.
    label : int
        True class number.
    pred : int
        Predicted class number.
    pred_one_hot : np.ndarray
        One-hot encoded prediction.
    """
    if image.shape == (784,):
        image = image.reshape((28, 28))
    _, axs = plt.subplots(1, 2)
    pred_one_hot = [[int(round(val * 100.0, 4)) for val in pred_one_hot[0]]]
    # Table data
    labels = [i for i in range(10)]
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=pred_one_hot, colLabels=labels, loc="center")
    # Image data
    axs[1].imshow(image, cmap=plt.get_cmap('gray_r'))
    # General plotting settings
    plt.figure_title('Label: %d, Pred: %d' % (label, pred))
    plt.show()


def display_convergence_error(
    train_losses: list,
    valid_losses: list
) -> None:
    """Display the convergence of the errors.

    Parameters
    ----------
    train_losses : list
        Train losses of the epochs.
    valid_losses : list
        Validation losses of the epochs
    """
    if len(valid_losses) > 0:
        plt.plot(len(train_losses), train_losses, color="red")
        plt.plot(len(valid_losses), valid_losses, color="blue")
        plt.legend(["Train", "Valid"])
    else:
        plt.plot(len(train_losses), train_losses, color="red")
        plt.legend(["Train"])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def display_convergence_acc(
    train_accs: list,
    valid_accs: list
) -> None:
    """Display the convergence of the accs.

    Parameters
    ----------
    train_accs : list
        Train accuracies of the epochs.
    valid_accs : list
        Validation accuracies of the epochs.
    """
    if len(valid_accs) > 0:
        plt.plot(len(train_accs), train_accs, color="red")
        plt.plot(len(valid_accs), valid_accs, color="blue")
        plt.legend(["Train", "Valid"])
    else:
        plt.plot(len(train_accs), train_accs, color="red")
        plt.legend(["Train"])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def plot_confusion_matrix(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    classes_list: list
) -> plt.figure:
    """Compute and create a plt.figure for the confusion matrix.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted classes.
    y_true : np.ndarray
        True classes.
    classes_list : list
        List of class names.

    Returns
    -------
    plt.figure
        Figure of the confusion matrix.
    """
    fig = plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes_list))
    plt.xticks(tick_marks, classes_list, rotation=45)
    plt.yticks(tick_marks, classes_list)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_to_image(
    fig: plt.figure
) -> tf.Tensor:
    """Plt plot/figure to tensorflow image.

    Parameters
    ----------
    fig : plt.figure
        Plt plot/figure.

    Returns
    -------
    tf.Tensor
        Tensorflow image object.
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    image = tf.io.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class ImageCallback(tf.keras.callbacks.Callback):
    """Custom tensorboard callback, to store images."""

    def __init__(
        self,
        model: tf.keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        log_dir: str = "./",
        classes_list: list = None,
        figure_fn: plt.Figure = None,
        figure_title: str = None,
    ):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        if classes_list is None:
            self.classes_list = [i for i in range(self.y_test[0])]
        else:
            self.classes_list = classes_list
        self.log_dir = log_dir
        img_file = os.path.join(self.log_dir, 'images')
        self.file_writer_images = tf.summary.create_file_writer(img_file)
        self.figure_fn = figure_fn
        if figure_title is None:
            self.figure_title = str(self.figure_fn)
        else:
            self.figure_title = figure_title

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict = None
    ):
        y_pred_prob = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        if self.figure_fn:
            fig = self.figure_fn(y_pred, y_true, self.classes_list)
            tf_image = plot_to_image(fig)
            figure_title_curr_epoch = self.figure_title + str(epoch)
            with self.file_writer_images.as_default():
                tf.summary.image(figure_title_curr_epoch, tf_image, step=epoch)


class ConfusionMatrix(ImageCallback):
    """Custom tensorbard callback, to store the confusion matrix figure."""

    def __init__(
        self,
        model: tf.keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        classes_list: list,
        log_dir: str
    ):
        self.figure_fn = plot_confusion_matrix
        self.figure_title = "Confusion Matrix"
        super().__init__(
            model,
            x_test,
            y_test,
            log_dir=log_dir,
            classes_list=classes_list,
            figure_fn=self.figure_fn,
            figure_title=self.figure_title,
        )

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict = None
    ):
        super().on_epoch_end(epoch, logs)
