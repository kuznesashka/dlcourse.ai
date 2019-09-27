import numpy as np

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = reg_strength * 2 * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
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

    mask_target = np.zeros(probabilities.shape)
    if probabilities.ndim == 1:
        mask_target[target_index] = 1
    elif target_index.ndim == 1:
        mask_target[tuple(np.arange(0, probabilities.shape[0])), tuple(target_index)] = 1
    else:
        mask_target[tuple(np.arange(0, probabilities.shape[0])), tuple(target_index.T[0])] = 1

    loss = -np.sum(mask_target * np.log(probabilities))

    dprediction = probabilities
    dprediction[mask_target.astype(bool)] = dprediction[mask_target.astype(bool)] - 1

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        result = np.copy(X)
        result[result < 0] = 0
        self.X = X
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_relu = np.zeros(d_out.shape)
        d_relu[self.X > 0] = 1
        d_result = d_out * d_relu

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        result = np.dot(X, self.W.value) + self.B.value
        self.X = X
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0)[:, np.newaxis].T
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
