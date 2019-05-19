import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# from cifar10_web import cifar10


def h(a):
    '''
    :return: The output of the activation function.
    '''
    return 1/(1+np.exp(-a))


def h_der(a):
    '''
    :return: The output of the activation function's derivative.
    '''
    return h(a) * (1 - h(a))


def softmax(s):
    '''
    Compute the softmax function.
    :param s: The softmax input.
    :return: The softmax output.
    '''
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))

    return exps/np.sum(exps, axis=1, keepdims=True)


def error(Y, T, w, lambda_reg):
    '''
    Return the error of the model at some given point.
    :param Y: The predicted values.
    :param T: The real values (has only one 1 at the spot where the label is True).
    :param w: A vector of all the weights unrolled.
    :param lambda_reg: The regularization term lambda.
    :return: The error.
    '''
    # c = 1e-50
    n_samples = T.shape[0]
    logp = np.log(Y[np.arange(n_samples), T.argmax(axis=1)])
    loss = (-1)*(1/n_samples)*(np.sum(logp) - 0.5*lambda_reg*np.linalg.norm(w)**2)

    return loss


class NeuralNetwork:
    def __init__(self, x, t, hidden_dim, output_dim, lr, epochs, lambda_reg=0.01, mini_batch_size=200):
        '''
        The constructor of the NeuralNetwork class.
        :param x: The dataset with examples in every row. Columns indicate the features.
        :param y: The correct labels for each example. y is an numpy array with one-hot-vectors in each row.
        :param hidden_dim: The number of the hidden units.
        :param output_dim: The number of different output values (e.g. for MNIST it is 10 for 10 digits).
        :param lr: The learning rate on which the model will learn.
        :param epochs: Number of epochs.
        :param lambda_reg: The lambda value for regularization. Small lambda will overfit and large lambda will underfit.
        :param mini_batch_size: The mini batch size for stochastic gradient descent. Recommended values 100 or 200
        '''
        self.x = x
        self.M = hidden_dim
        self.lr = lr
        in_dim = x.shape[1]
        out_dim = output_dim

        self.batch_size = mini_batch_size
        self.epochs = epochs

        # Initialize weights
        self.w1 = np.random.randn(in_dim + 1, self.M) * np.sqrt(1/self.M)
        # Add bias to W1
        self.w1[0, :] = np.zeros((1, self.M))

        self.w2 = np.random.randn(self.M + 1, out_dim) * np.sqrt(1/out_dim)
        # Add bias to W2
        self.w2[0, :] = np.zeros((1, out_dim))

        # Learning rate for gradient descent
        self.lr = lr
        # Regularization term lambda
        self.lambda_reg = lambda_reg

        # True values
        self.t = t

        # Initialize helper matrices
        self.prediction = None
        self.zeta_in = None
        self.zeta = None
        self.x_with_bias = None

        print(f'Shape of X: {self.x.shape}')
        print(f'Shape of T: {self.t.shape}')
        print(f'Shape of W1: {self.w1.shape}')
        print(f'Shape of W2: {self.w2.shape}')
        print(f'Shape of b1: {self.w1[0, :].shape}')
        print(f'Shape of b2: {self.w2[0, :].shape}')

    def feedforward(self):
        '''
        The feed forward method for 'predicting' the output of the examples given in self.x.
        Update the prediction for each epoch
        '''
        # Forward for w1
        # The x matrix after adding a bias vector as a new column.
        self.x_with_bias = np.ones((self.x.shape[0], self.x.shape[1] + 1))
        self.x_with_bias[:, 1:] = self.x  # N x D+1

        self.zeta_in = np.dot(self.x_with_bias, self.w1)  # N x M
        self.zeta = np.ones((self.x.shape[0], self.zeta_in.shape[1]+1))  # N x M+1
        self.zeta[:, 1:self.zeta.shape[1]] = h(self.zeta_in)
        soft_in = np.dot(self.zeta, self.w2)  # N x K

        self.prediction = softmax(soft_in)  # N x K

    def backprop(self, epoch, gradient_check=False):
        '''
        Perform backpropagation using gradient descent.
        :param epoch: The current epoch of training.
        :param gradient_check: If true then perform gradient checking, else just backpropagate the error in order to maximize the error function.
        :return: Update the weights.
        '''
        # Weights before the update
        weights = np.concatenate((self.w1.ravel(), self.w2.ravel()), axis=0)

        n = self.x.shape[0]

        loss = error(self.prediction, self.t, weights, self.lambda_reg)
        print(f'Epoch {epoch} / {self.epochs}, Error: {loss}')

        Delta1 = np.zeros(self.w1.shape)
        Delta2 = np.zeros(self.w2.shape)

        # d3 = np.zeros((bs, t_i.shape[1]))  # bs x K
        d3 = self.prediction - self.t

        # d2 = np.zeros((bs, self.w1.shape[1]))  # bs x M

        temp = np.dot(self.w2, d3.T)  # M+1 x bs
        d2 = np.transpose(temp[1:, :] * h_der(self.zeta_in).T)  # bs x M

        Delta1 = Delta1 + np.transpose(np.dot(d2.T, self.x_with_bias))  # D+1 x M
        Delta2 = Delta2 + np.transpose(np.dot(d3.T, self.zeta))  # M+1 x K

        # Update gradients
        # w1_grad = np.zeros(self.w1.shape)
        # w2_grad = np.zeros(self.w2.shape)

        w1_grad = Delta1/n + (self.lambda_reg/n) * self.w1
        w2_grad = Delta2/n + (self.lambda_reg/n) * self.w2

        self.w1 -= self.lr * w1_grad
        self.w2 -= self.lr * w2_grad

        # Gradient checking
        if gradient_check:
            w_grad = np.concatenate((w1_grad.ravel(), w2_grad.ravel()), axis=0)
            self.grad_check(1e-6, w_grad, weights)

    def grad_check(self, epsilon, w_grad, weights):
        '''
        Perform gradient checking for the gradients we calculated during back propagation.
        :param epsilon: The value at which we will calculate the differences for the gradient approximates.
        :param w_grad: The gradients computed in backpropagation.
        :param weights: A vector of all the weights.
        '''

        print('Checking Gradients...')
        w_grad_approx = np.zeros(weights.shape[0])
        for j in range(len(weights)):
            new_weights = weights.copy()
            new_weights[j] = weights[j] + epsilon

            # Get new weights
            W1 = np.reshape(new_weights[:self.w1.shape[1]*(self.w1.shape[0])], (self.w1.shape[0], self.w1.shape[1]))
            W2 = np.reshape(new_weights[self.w1.shape[1]*(self.w1.shape[0]):], (self.w2.shape[0], self.w2.shape[1]))

            zeta_in = np.dot(self.x_with_bias, W1)  # N x M
            zeta = np.ones((self.x.shape[0], zeta_in.shape[1]+1))  # N x M+1
            zeta[:, 1:zeta.shape[1]] = h(zeta_in)
            soft_in = np.dot(zeta, W2)  # N x K

            pred = softmax(soft_in)  # N x K

            loss1 = error(pred, self.t, new_weights, self.lambda_reg)

            # Minus epsilon
            # new_weights_minus = weights.copy()
            # new_weights_minus[j] = weights[j] - epsilon
            new_weights[j] = new_weights[j] - 2*epsilon

            W1n = np.reshape(new_weights[:self.w1.shape[1]*(self.w1.shape[0])], (self.w1.shape[0], self.w1.shape[1]))
            W2n = np.reshape(new_weights[self.w1.shape[1]*(self.w1.shape[0]):], (self.w2.shape[0], self.w2.shape[1]))

            zeta_in = np.dot(self.x_with_bias, W1n)  # N x M
            zeta = np.ones((self.x.shape[0], zeta_in.shape[1]+1))  # N x M+1
            zeta[:, 1:zeta.shape[1]] = h(zeta_in)
            soft_in = np.dot(zeta, W2n)  # N x K

            pred = softmax(soft_in)  # N x K

            loss2 = error(pred, self.t, new_weights, self.lambda_reg)

            # Calculate gradient approximation
            w_grad_approx[j] = (loss1-loss2)/(2*epsilon)

        # Calculate relative error
        rel_err_w = np.linalg.norm(w_grad_approx - w_grad)/(np.linalg.norm(w_grad) + np.linalg.norm(w_grad_approx))
        print()
        # Set 10^(-7) as a boundary for failure
        if rel_err_w < 1e-7:
            print('Gradient checking succeeded!')
        else:
            print('Gradient checking failed.')
        print(f'Relevant different from using gradient check: {rel_err_w}')
        print()

    def predict(self, data):
        '''
        Make a prediction based on the data given.
        :param data: A vector with image pixels.
        :return: The prediction of our neural network.
        '''
        # Make explicit that x is an array so we will then be able to add bias
        self.x = np.ones((1, len(data)))
        self.x[0, :] = data
        self.feedforward()
        return self.prediction.argmax()


def train(model, epochs, grad_checking=False):
    '''
    Train an already initialized model.
    :param model: A NeuralNetwork instance.
    :param epochs: Number of epochs to train the model.
    :param grad_checking: True if needed to perform gradient checking to check gradient differences.
    :return: The trained model.
    '''
    for epoch in range(1, epochs+1):
        model.feedforward()
        model.backprop(epoch, grad_checking)


def get_acc(model, x, t):
    '''
    Get the accuracy of our model for the set X containing the examples.
    :param model: The model that we will get the accuracy for
    :param x: The dataset containing the examples.
    :param t: The set containing the true labels for each example.
    :return: The accuracy of our model's prediction for the dataset given.
    '''
    acc = 0
    for xx, tt in zip(x, t):
        s = model.predict(xx)
        if s == np.argmax(tt):
            acc += 1
    return acc/len(x)*100


def save_model(model_to_save, filename='nn_model', save_path='.'):
    '''
    Save the model trained using pickle.
    :param model_to_save: The model that will be saved.
    :param filename: The filename - default is nn_model.
    :param save_path: The path where the model will be saved.
    '''
    with open((save_path + '//' + filename + '.pickle'), 'wb') as f:
        pickle.dump(model_to_save, f)


def load_model(filename='nn_model', path='.'):
    '''
    Load already pre-trained model
    :param filename: The name of the NN model file.
    :param path: The path to the file.
    :return: The pre-trained model loaded.
    '''
    with open((path + '//' + filename + '.pickle'), 'rb') as f:
        loaded_model = pickle.load(f)
        return loaded_model


if __name__ == '__main__':
    # Uncomment the following to use the load_digits dataset from sklearn
    print('Using the load_digits dataset from sklearn!')
    dig = load_digits()
    onehot_target = pd.get_dummies(dig.target)
    x_train, x_test, t_train, t_test = train_test_split(dig.data, onehot_target, test_size=0.2, random_state=20)
    x_max = 16

    # Choose the CIFAR-10 dataset. If the below is uncommented then the cifar_10_web library will download the
    # cifar-10 dataset in the path specified (or in C:\\Users\\user\\data\\cifar10\\ by default if path is None)

    # path = 'C:\\Users\\user\\data\\cifar10\\'
    # print('Getting CIFAR-10 dataset...')
    # x_train, t_train, x_test, t_test = cifar10(path=None)
    # x_max = np.max(x_train)
    # print('Successfully read CIFAR-10 dataset!')

    # Initialize network parameters
    M = 200
    K = t_train.shape[1]
    lr = 0.5
    mini_batch_size = 100
    # Small lambda_reg will overfit the dataset
    lambda_reg = 0.01
    # Define number of epochs to train the network
    epochs = 80

    # import time
    # start = time.time()

    # Initialize the model
    model = NeuralNetwork(x_train/x_max, np.array(t_train), M, K, lr, epochs, lambda_reg, mini_batch_size)

    # Boolean for if you want to perform gradient checking
    grad_checking = False
    # Train the Neural Network model
    train(model, epochs, grad_checking)

    # Save model trained and then load it using pickle.
    save_model(model)
    model = load_model()

    # Print statistics
    print("Training accuracy: ", get_acc(model, x_train/x_max, np.array(t_train)))
    print("Test accuracy: ", get_acc(model, x_test/x_max, np.array(t_test)))
    # print("Time taken: " + str(time.time() - start))
