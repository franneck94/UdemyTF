import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def display_digit(image: np.ndarray, label: np.ndarray = None, pred_label: np.ndarray = None) -> None:
    """Display the MNIST image.
    If the `label` and `pred_label` is given, these are also displayed.

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
        plt.figure_title('Label: %d' % (label))
    elif label is not None:
        plt.figure_title('Label: %d, Pred: %d' % (label, pred_label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_digit_and_predictions(image: np.ndarray, label: int, pred: int, pred_one_hot: np.ndarray) -> None:
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


def display_convergence_error(train_losses: list, valid_losses: list) -> None:
    """Display the convergence of the errors.

    Parameters
    ----------
    train_losses : list
        Train losses of the epochs.
    valid_losses : list
        Validation losses of the epochs
    """
    if len(valid_losses) > 0:
        plt.plot(range(len(train_losses)), train_losses, color="red")
        plt.plot(range(len(valid_losses)), valid_losses, color="blue")
        plt.legend(["Train", "Valid"])
    else:
        plt.plot(range(len(train_losses)), train_losses, color="red")
        plt.legend(["Train"])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def display_convergence_acc(train_accs: list, valid_accs: list) -> None:
    """Display the convergence of the accs.

    Parameters
    ----------
    train_accs : list
        Train accuracies of the epochs.
    valid_accs : list
        Validation accuracies of the epochs.
    """
    if len(valid_accs) > 0:
        plt.plot(range(len(train_accs)), train_accs, color="red")
        plt.plot(range(len(valid_accs)), valid_accs, color="blue")
        plt.legend(["Train", "Valid"])
    else:
        plt.plot(range(len(train_accs)), train_accs, color="red")
        plt.legend(["Train"])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, classes_list: list) -> plt.figure:
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
