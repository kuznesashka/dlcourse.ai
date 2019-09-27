import numpy as np
from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization

class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.RELU_1 = ReLULayer()
        self.RELU_2 = ReLULayer()
        self.FullyConnected_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.FullyConnected_2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        y1 = self.FullyConnected_1.forward(X)
        y2 = self.RELU_1.forward(y1)
        y3 = self.FullyConnected_2.forward(y2)
        y_result = self.RELU_2.forward(y3)
        loss, d_out1 = softmax_with_cross_entropy(y_result, y)

        d_out2 = self.RELU_2.backward(d_out1)
        d_out3 = self.FullyConnected_2.backward(d_out2)
        dW2 = self.FullyConnected_2.params()['W'].grad
        dB2 = self.FullyConnected_2.params()['B'].grad

        d_out4 = self.RELU_1.backward(d_out3)
        d_out_result = self.FullyConnected_1.backward(d_out4)
        dW1 = self.FullyConnected_1.params()['W'].grad
        dB1 = self.FullyConnected_1.params()['B'].grad

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_l1, dW1_l = l2_regularization(self.FullyConnected_1.params()['W'].value, self.reg)
        loss_l2, dW2_l = l2_regularization(self.FullyConnected_2.params()['W'].value, self.reg)
        loss_l3, dB1_l = l2_regularization(self.FullyConnected_1.params()['B'].value, self.reg)
        loss_l4, dB2_l = l2_regularization(self.FullyConnected_2.params()['B'].value, self.reg)

        self.FullyConnected_1.params()['W'].grad = dW1 + dW1_l
        self.FullyConnected_2.params()['W'].grad = dW2 + dW2_l
        self.FullyConnected_1.params()['B'].grad = dB1 + dB1_l
        self.FullyConnected_2.params()['B'].grad = dB2 + dB2_l

        return loss+loss_l1+loss_l2+loss_l3+loss_l4

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused

        y1 = self.FullyConnected_1.forward(X)
        y2 = self.RELU_1.forward(y1)
        y3 = self.FullyConnected_2.forward(y2)
        predictions = self.RELU_2.forward(y3)

        if predictions.ndim == 1:
            predictions_new = predictions - np.max(predictions)
        else:
            maximum = np.max(predictions, axis=1)
            predictions_new = predictions - maximum[:, np.newaxis]
        predictions_new = np.exp(predictions_new)
        predictions_sum = np.sum(predictions_new, axis=(predictions.ndim - 1))
        if predictions.ndim == 1:
            probabilities = predictions_new / predictions_sum
        else:
            probabilities = predictions_new / predictions_sum[:, np.newaxis]

        pred = np.argmax(probabilities, axis=1)

        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        return {'W1': self.FullyConnected_1.params()['W'], 'W2': self.FullyConnected_2.params()['W'],
                'B1': self.FullyConnected_1.params()['B'], 'B2': self.FullyConnected_2.params()['B']}

