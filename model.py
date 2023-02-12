import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib as plt
from loss import CrossEntropyLoss


# the model class ======================================================================
class ConvolutionalNetwork:
    def __init__(self):
        self.layers = None
        self.loss = CrossEntropyLoss()

    # takes a list of layers as input
    def build_netword(self, layers):
        self.layers = layers

    def forward(self, input, tq=False):
        output = input
        if tq:
            for layer in tqdm(self.layers, desc='Forward pass', position=0, leave=True):
                output = layer.forward(output)
        else:
            for layer in self.layers:
                output = layer.forward(output)
        return output
    
    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        
    def train(self, X_train, y_train, X_test, y_test, learning_rate, epochs, batch_size):
        metric_data = []
        # mini-batch gradient descent
        n_batches = len(X_train) // batch_size
        for epoch in tqdm(range(epochs), desc='Training with ', position=0, leave=True):
            for i in tqdm(range(n_batches), desc='Epoch {}'.format(epoch), position=1, leave=True):
                # get batch
                X_batch = X_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]
                
                output = self.forward(X_batch)
                # print(output.shape, y_batch.shape)
                loss = self.loss.forward(output, y_batch)
                output_gradient = self.loss.backward()
                self.backward(output_gradient, learning_rate)
            # evaluate after each epoch
            accuracy, validation_loss, macro_f1 = self.evaluate(X_test, y_test)
            metric_data.append([epoch, accuracy, validation_loss, macro_f1, loss])
            print('Epoch: {}, Training Loss: {}'.format(epoch, loss))

        self.plot_metrics(metric_data)
        self.plot_confustion(X_test, y_test)


    def predict(self, X, tq=False):
        output = self.forward(X, tq=tq)
        # get index of max value in each row:
        predictions = np.argmax(output, axis=1)
        return predictions
    
    def plot_confustion(self, X, y):
        predictions = self.predict(X)
        # convert y from one hot encoding to single value:
        y = np.argmax(y, axis=1)
        
        # use sklearn for metrics
        confusion_matrix = metrics.confusion_matrix(y, predictions)

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4,5,6,7,8,9]
        )
        cm_display.plot()
        plt.savefig('confusion.png')
        plt.clf()       

    def evaluate(self, X, y, tq=False):
        predictions = self.predict(X, tq=tq)
        # convert y from one hot encoding to single value:
        y = np.argmax(y, axis=1)
        
        # use sklearn for metrics
        validation_loss = self.loss.forward(predictions, y)
        accuracy = metrics.accuracy_score(y, predictions)
        macro_f1 = metrics.f1_score(y, predictions, average='macro')
        print('Accuracy: {}, Val_loss: {}, Macro_f1: {}'.format(accuracy, validation_loss, macro_f1))
        return accuracy, validation_loss, macro_f1
    
    def plot_metrics(self, metric_data):
        epochs = [i[0] for i in metric_data]
        accuracy = [i[1] for i in metric_data]
        val_loss = [i[2] for i in metric_data]
        macro_f1 = [i[3] for i in metric_data]
        loss = [i[4] for i in metric_data]

        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        ax[0, 0].plot(epochs, accuracy)
        ax[0, 0].set_title('Accuracy with Validation Set')
        ax[0, 0].set_ylabel('Accuracy')
        ax[0, 0].set_xlabel('Epochs')

        ax[0, 1].plot(epochs, val_loss)
        ax[0, 1].set_title('Validation Loss')
        ax[0, 1].set_ylabel('Loss')
        ax[0, 1].set_xlabel('Epochs')

        ax[1, 0].plot(epochs, macro_f1)
        ax[1, 0].set_title('Macro F1 with Validation Set')
        ax[1, 0].set_ylabel('Macro F1')
        ax[1, 0].set_xlabel('Epochs')

        ax[1, 1].plot(epochs, loss)
        ax[1, 1].set_title('Training Loss')
        ax[1, 1].set_ylabel('Loss')
        ax[1, 1].set_xlabel('Epochs')
        plt.savefig('metrics.png')
        plt.clf()
# end of model class ======================================================================