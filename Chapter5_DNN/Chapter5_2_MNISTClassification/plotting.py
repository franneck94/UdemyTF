# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display the Digit from the image
# If the Label and PredLabel is given display it too
def display_digit(image, label=None, pred_label=None):
    if image.shape == (784,):
        image = image.reshape((28, 28))
    label = np.argmax(label, axis=0)
    if pred_label is None and label is not None:
        plt.title('Label: %d' % (label))
    elif label is not None:
        plt.title('Label: %d, Pred: %d' % (label, pred_label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_digit_and_predictions(image, label, pred, pred_one_hot):
    if image.shape == (784,):
        image = image.reshape((28, 28))
    fig, axs =plt.subplots(1,2)
    pred_one_hot = [[int(round(val * 100.0, 4)) for val in pred_one_hot[0]]]
    # Table data
    labels = [i for i in range(10)]
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].table(cellText=pred_one_hot, colLabels=labels, loc="center")
    # Image data
    axs[1].imshow(image, cmap=plt.get_cmap('gray_r'))
    # General plotting settings
    plt.title('Label: %d, Pred: %d' % (label, pred))
    plt.show()

# Display the convergence of the errors
def display_convergence_error(train_losses, valid_losses):
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

# Display the convergence of the accs
def display_convergence_acc(train_accs, valid_accs):
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

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()