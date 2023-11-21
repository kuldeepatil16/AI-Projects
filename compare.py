import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

class MyNeuralNetwork:
    def __init__(self, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage):
        self.L = num_layers
        self.n = num_units
        self.num_epochs = num_epochs
        self.eta = learning_rate
        self.alpha = momentum
        self.fact = activation_function
        self.validation_percentage = validation_percentage
        self.xi = [np.zeros(layer_units) for layer_units in num_units]
        self.h = [np.zeros(layer_units) for layer_units in num_units]
        self.w = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
        self.theta = [np.zeros(layer_units) for layer_units in num_units]
        self.delta = [np.zeros(layer_units) for layer_units in num_units]
        self.d_w = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
        self.d_theta = [np.zeros(layer_units) for layer_units in num_units]
        self.d_w_prev = [None] + [np.zeros((num_units[i], num_units[i - 1])) for i in range(1, num_layers)]
        self.d_theta_prev = [np.zeros(layer_units) for layer_units in num_units]
        self.training_error = []
        self.validation_error = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - x**2

    def activation(self, x):
        if self.fact == 'sigmoid':
            return self.sigmoid(x)
        elif self.fact == 'relu':
            return self.relu(x)
        elif self.fact == 'linear':
            return self.linear(x)
        elif self.fact == 'tanh':
            return self.tanh(x)

    def activation_derivative(self, x):
        if self.fact == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.fact == 'relu':
            return self.relu_derivative(x)
        elif self.fact == 'linear':
            return self.linear_derivative(x)
        elif self.fact == 'tanh':
            return self.tanh_derivative(x)

    def feed_forward(self, sample):
        self.xi[0] = sample
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self.activation(self.h[l])

    def backpropagate(self, target):
        self.delta[self.L - 1] = self.activation_derivative(self.xi[self.L - 1]) * (self.xi[self.L - 1] - target)
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = self.activation_derivative(self.xi[l]) * np.dot(self.w[l + 1].T, self.delta[l + 1])

    def update_weights(self):
        for l in range(1, self.L):
            self.d_w[l] = -self.eta * np.outer(self.delta[l], self.xi[l - 1]) + self.alpha * self.d_w_prev[l]
            self.d_theta[l] = self.eta * self.delta[l] + self.alpha * self.d_theta_prev[l]
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]

    def calculate_total_error(self, X, y):
        total_error = 0.0
        for i in range(X.shape[0]):
            self.feed_forward(X[i])
            total_error += 0.5 * np.sum((self.xi[self.L - 1] - y[i]) ** 2)
        return total_error

    def fit(self, X, y):
        n_samples = X.shape[0]
    
        if self.validation_percentage > 0:
           n_train = int(n_samples * (1.0 - self.validation_percentage))
           X_train, y_train = X[:n_train], y[:n_train]
           X_val, y_val = X[n_train:], y[n_train:]
        else:
           X_train, y_train = X, y
           X_val, y_val = np.array([]), np.array([])

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_val shape:", X_val.shape)
        print("y_val shape:", y_val.shape)

        for epoch in range(self.num_epochs):
            for i in range(X_train.shape[0]):
                sample = X_train[i]
                target = y_train[i]

                self.feed_forward(sample)
                self.backpropagate(target)
                self.update_weights()

            train_error = self.calculate_total_error(X_train, y_train)
            self.training_error.append(train_error)

            if X_val.shape[0] > 0:
               val_error = self.calculate_total_error(X_val, y_val)
               self.validation_error.append(val_error)


    def predict(self, X):
        predictions = []
        for sample in X:
            self.feed_forward(sample)
            predictions.append(self.xi[self.L - 1][:, 0].copy())  # Extract the first column as a 1D array
        return np.array(predictions)

    def loss_epochs(self):
        return np.column_stack((self.training_error, self.validation_error))

# Function to train a neural network on the given datasets 
def train_neural_network(dataset, input_columns, output_column, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage):
    X = dataset[input_columns].values
    y = dataset[output_column].values.reshape(-1, 1)  # Ensure y is a 2D array

    nn = MyNeuralNetwork(num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    nn.fit(X, y)
    
    X_test = X  # Using the same dataset for testing
    predictions = nn.predict(X_test)

    print("Shape of y:", y.shape)
    print("Shape of predictions:", predictions.shape)
    mape = mean_absolute_percentage_error(y.flatten(), predictions.flatten())

    return nn, predictions, mape


# Load the synthetic dataset
dataset = pd.read_csv('A1-synthetic.csv')

# Defining input and output columns for the synthetic dataset
input_columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
output_column = 'z'

# Defining parameter configurations to explore
num_layers_list = [3, 4, 5]
num_units_list = [[len(input_columns)] + [5, 2, 1], [len(input_columns)] + [10, 5, 1], [len(input_columns)] + [8, 4, 2, 1]]
num_epochs_list = [500, 1000]
learning_rate_list = [0.01, 0.1, 0.2]
momentum_list = [0.0, 0.2, 0.4]
activation_function_list = ["sigmoid", "relu"]

# Store results in a DataFrame
results_columns = ['NumLayers', 'NumUnits', 'NumEpochs', 'LearningRate', 'Momentum', 'ActivationFunction', 'MAPE']
results_df = pd.DataFrame(columns=results_columns)

# Iterate through different parameter combinations
for num_layers in num_layers_list:
    for num_units in num_units_list:
        for num_epochs in num_epochs_list:
            for learning_rate in learning_rate_list:
                for momentum in momentum_list:
                    for activation_function in activation_function_list:
                        validation_percentage = 0.2 #Train the neural network
                        nn, predictions, mape = train_neural_network(dataset, input_columns, output_column, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
                        
                        # Store results in the DataFrame
                        results_df = results_df.append({
                            'NumLayers': num_layers,
                            'NumUnits': num_units,
                            'NumEpochs': num_epochs,
                            'LearningRate': learning_rate,
                            'Momentum': momentum,
                            'ActivationFunction': activation_function,
                            'MAPE': mape
                        }, ignore_index=True)

# Display the results table
print(results_df)

# Scatter Plots
min_mape_row = results_df.loc[results_df['MAPE'].idxmin()]
min_mape_params = {
    'NumLayers': int(min_mape_row['NumLayers']),
    'NumUnits': min_mape_row['NumUnits'],
    'NumEpochs': int(min_mape_row['NumEpochs']),
    'LearningRate': min_mape_row['LearningRate'],
    'Momentum': min_mape_row['Momentum'],
    'ActivationFunction': min_mape_row['ActivationFunction']
}

# Train the neural network with the parameters that give the minimum MAPE
nn_min_mape, predictions_min_mape, _ = train_neural_network(dataset, input_columns, output_column, **min_mape_params)

# Plot scatter plots for some rows in the table
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, ax in enumerate(axes.flatten()):
    sample_index = i * 10  # Adjust this to choose different rows
    ax.scatter(dataset[output_column].iloc[sample_index], predictions_min_mape[sample_index])
    ax.set_title(f'Sample {sample_index + 1}')
    ax.set_xlabel('Real Values')
    ax.set_ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# Error Evolution Plots
loss_data_min_mape = nn_min_mape.loss_epochs()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axes[0].plot(loss_data_min_mape[:, 0], label='Training Error')
axes[0].set_title('Training Error Evolution')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Error')
axes[0].legend()

axes[1].plot(loss_data_min_mape[:, 1], label='Validation Error')
axes[1].set_title('Validation Error Evolution')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Error')
axes[1].legend()

plt.tight_layout()
plt.show()