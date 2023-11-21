# AI-Projects
Neural Network Predictions with Back-Propagation and Linear Regression
Programming of a neural network with back propagation from scratch.
The code must be able to deal with arbitrary multilayer networks. For example, a network with architecture 3:9:5:1 (4 layers, 3 input units, 1 output unit, and two hidden layers with 9 and 5 units, respectively), would have n=[3; 9; 5; 1], and xi would be an array of length 4 (one component
per layer), with xi[1] and array of real numbers of length 3, xi[2] and array of real numbers of length 9, xi[3] and array of real numbers of length 5, and xi[4] and array of real numbers of length 1. Similarly, w[2] would be an array 9x3, w[3] an array 5x9, and w[4] and array 1x5; w[1] is not used.
The code will receive one input dataset, and using the percentage of data that is passed as a parameter in the class constructor, should divide this dataset into training and validation. If the percentage is 0, then we consider that there is no validation, and all the input data is used for training.
It also includes comparing predictions using the three models (BP, BP-F, MLR-F).
