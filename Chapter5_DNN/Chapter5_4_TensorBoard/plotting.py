import io
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf

def display_digit(image, label=None, pred_label=None):
    """Display the Digit from the image.
    If the Label and PredLabel is given, display it too.
    """
    if image.shape == (784,):
        image = image.reshape((28, 28))
    label = np.argmax(label, axis=0)
    if pred_label is None and label is not None:
        plt.figure_title('Label: %d' % (label))
    elif label is not None:
        plt.figure_title('Label: %d, Pred: %d' % (label, pred_label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_digit_and_predictions(image, label, pred, pred_one_hot):
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

def display_convergence_error(train_losses, valid_losses):
    """Display the convergence of the errors.
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

def display_convergence_acc(train_accs, valid_accs):
    """Display the convergence of the accs
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

def plot_confusion_matrix(y_pred, y_true, classes_list):
    """Compute and create a plt.figure for the confusion matrix.
    """
    fig = plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y_pred, y_true)
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
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def plot_to_image(fig):
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test, log_dir="./", classes_list=None, figure_fn=None, figure_title=None):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        if classes_list is None:
            self.classes_list = [i for i in range(self.y_test[0])]
        else:
            self.classes_list = classes_list
        self.log_dir = log_dir
        self.file_writer_images = tf.summary.create_file_writer(os.path.join(self.log_dir, 'images'))
        self.figure_fn = figure_fn
        if figure_title is None:
            self.figure_title = str(self.figure_fn)
        else:
            self.figure_title = figure_title

    def on_epoch_end(self, epoch, logs):
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
    def __init__(self, model, x_test, y_test, classes_list, log_dir):
        self.figure_fn = plot_confusion_matrix
        self.figure_title = "Confusion Matrix"
        super().__init__(model, x_test, y_test, log_dir=log_dir, classes_list=classes_list, figure_fn=self.figure_fn, figure_title=self.figure_title)

    def on_epoch_end(self, epoch, logs):
        super().on_epoch_end(epoch, logs)
