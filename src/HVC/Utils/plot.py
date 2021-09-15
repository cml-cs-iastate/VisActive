import os
import matplotlib.pyplot as plt


# Tested
def plot_loss(history, output_directory, model_loss_file='model_loss.png'):
    """
    plot the classification accuracy and loss
    :param history: the dictionary of the validation and Training results
    :param output_directory:
    :param model_loss_file: the file name of model loss plot
    """
    # list all data in history
    print('History Metrics:', history.keys())

    # summarize history for loss
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training loss'], loc='upper left')
    plt.savefig(os.path.join(output_directory, model_loss_file), dpi=300)
    plt.clf()


# Tested
def plot_loss_accuracy(history, output_directory, model_acc_file='model_accuracy.png', model_loss_file='model_loss.png'):
    """
    plot the classification accuracy and loss
    :param history: the dictionary of the validation and Training results
    :param output_directory:
    :param model_acc_file: the file name of model accuracy plot
    :param model_loss_file: the file name of model loss plot
    """
    # list all data in history
    print('History Metrics:', history.keys())

    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_directory, model_acc_file), dpi=300)
    # plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_directory, model_loss_file), dpi=300)
    # plt.show()
    plt.clf()
