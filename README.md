# SimpleNeuralNet
A simple neural network with 1 hidden layer in python using gradient descent.

Consider normalizing the input matrix to [0, 1] before initiliazing the NN. 
T is the matrix of the true labels and Y is the matrix of our predictions (this may cause confusion).
The activation function is the sigmoid.

In the NeuralNetwork class a method for gradient checking exists in order to validate your results. A normal threshold is 1e-6 which I also use.

Tune the parameters in order to get a better result. Note that a greater value of M can improve the model's accuracy. If using a larger dataset(than load_digits) then lower the learning rate and increase the epochs in order to get a better result.
