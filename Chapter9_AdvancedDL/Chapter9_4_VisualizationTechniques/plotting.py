# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow.keras.backend as K

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
def display_convergence_error(train_error, test_error):
    plt.plot(train_error, color="red")
    plt.plot(test_error, color="blue")
    plt.legend(["Train", "Test"])
    plt.title('Error of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

# Display the convergence of the accs
def display_convergence_acc(train_acc, test_acc):
    plt.plot(train_acc, color="red")
    plt.plot(test_acc, color="blue")
    plt.legend(["Train", "Test"])
    plt.title('Accs of the NN')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
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

##############################################
###########    VISUALIZATIONS   ##############
##############################################

def get_occlusion(img, label, box_size, step_size, model):
    rows, cols, depth = img.shape
    occulsion_map = np.full(shape=(rows, cols), fill_value=1.0)
    box = np.full(shape=(box_size, box_size, depth), fill_value=0.0)
    true_class_idx = np.argmax(label)

    for i in range(0, rows-box_size+1, step_size):
        for j in range(0, cols-box_size+1, step_size):
            img_with_box = img.copy()
            img_with_box[i:i+box_size, j:j+box_size] = box
            y_pred = model.predict(img_with_box.reshape((1, rows, cols, depth)))[0]
            prob_right_class = y_pred[true_class_idx]
            occulsion_map[i:i+step_size, j:j+step_size] = \
                np.full(shape=(step_size, step_size), fill_value=prob_right_class)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    cmap = "Spectral"
    ax1.imshow(img)
    heatmap = ax2.imshow(occulsion_map, cmap=cmap)
    cbar = plt.colorbar(heatmap)
    plt.show()

def get_heatmap(img, model):
    rows, cols, depth = img.shape
    heatmap_layers = [layer for layer in model.layers if "heatmap" in layer.name]

    for layer_index, heatmap_layer in enumerate(heatmap_layers):
        heatmap_output = K.function([model.layers[0].input], [heatmap_layer.output])
        heatmap_output = heatmap_output([img.reshape(1, rows, cols, depth)])[0]

        heatmap = np.squeeze(heatmap_output, axis=0)
        heatmap = np.transpose(heatmap, axes=(2, 0, 1))
        num_subplots = 16
        subplot_shape = (4, 4)
        plt.figure(num=1, figsize=(10, 10))

        for filter_index, heatmap_filter in enumerate(heatmap[:num_subplots]):
            plt.subplot(subplot_shape[0], subplot_shape[1], filter_index +1)
            plt.title("Filter: " + str(filter_index + 1) + " of Layer: " + str(layer_index))
            plt.imshow(heatmap_filter)
        
        plt.tight_layout()
        plt.show()
