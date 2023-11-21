import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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
        self.w = [None] + [np.random.randn(num_units[i], num_units[i - 1] + 1) for i in range(1, num_layers)]
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
        errors = []
        for i in range(X.shape[0]):
            self.feed_forward(X[i])
            error = 0.5 * np.sum((self.xi[self.L - 1] - y[i]) ** 2)
            errors.append(error)
        return errors

    def fit(self, X, y):
        n_samples = X.shape[0]
        if self.validation_percentage > 0:
            n_train = int(n_samples * (1.0 - self.validation_percentage))
            X_train = X[:n_train]
            y_train = y[:n_train]
            X_val = X[n_train:]
            y_val = y[n_train:]
        else:
            X_train = X
            y_train = y
            X_val = np.array([])
            y_val = np.array([])

        for epoch in range(self.num_epochs):
            epoch_errors = []
            
            # Shuffle the training data for each epoch
            indices = np.random.permutation(n_train)
            X_train = X_train.iloc[indices]  # Use iloc to select rows based on integer indices
            y_train = y_train[indices]

            for i in range(n_train):
                sample = X_train.iloc[i].values  # Access the values of the DataFrame row
                target = y_train[i]

                self.feed_forward(sample)
                self.backpropagate(target)
                self.update_weights()

            train_errors = self.calculate_total_error(X_train.values, y_train)
            epoch_errors.append(np.mean(train_errors))

            if X_val.shape[0] > 0:
                val_errors = self.calculate_total_error(X_val.values, y_val)
                epoch_errors.append(np.mean(val_errors))
            else:
                epoch_errors.append(np.nan)

            self.training_error.append(np.mean(train_errors))
            self.validation_error.append(np.mean(val_errors) if X_val.shape[0] > 0 else np.nan)

            max_len = min(len(self.training_error), len(self.validation_error), len(epoch_errors))
            epoch_errors += [np.nan] * (max_len - len(epoch_errors))
            self.validation_error += [np.nan] * (max_len - len(self.validation_error))

        max_len = min(len(self.training_error), len(self.validation_error))
        loss_data = np.column_stack((list(range(max_len)), self.training_error[:max_len], self.validation_error[:max_len]))
        np.savetxt("loss_data_dataset1.csv", loss_data, delimiter=",")

    def predict(self, X):
        predictions = []
        for sample in X:
            self.feed_forward(sample)
            predictions.append(self.xi[self.L - 1].copy())
        return np.array(predictions)

    def loss_epochs(self):
        return np.column_stack((self.training_error, self.validation_error))
    
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def explore_parameters(dataset, input_columns, output_column, y, num_epochs_list, learning_rate_list, momentum_list, activation_function_list, validation_percentage_list):
    results = []

    for num_layers in [3, 4, 5]:  # Modify as needed
        for num_units in [[len(input_columns)] + [10, 5, 1], [len(input_columns)] + [15, 10, 5, 1]]:  # Modify as needed
            for num_epochs in num_epochs_list:
                for learning_rate in learning_rate_list:
                    for momentum in momentum_list:
                        for activation_function in activation_function_list:
                            for validation_percentage in validation_percentage_list:
                                nn = MyNeuralNetwork(
                                    num_layers, num_units, num_epochs,
                                    learning_rate, momentum, activation_function,
                                    validation_percentage
                                )

                                nn.fit(dataset, y)

                                predictions = nn.predict(dataset)

                                mape = calculate_mape(y, predictions)
                                results.append({
                                    'num_layers': num_layers,
                                    'num_units': num_units,
                                    'num_epochs': num_epochs,
                                    'learning_rate': learning_rate,
                                    'momentum': momentum,
                                    'activation_function': activation_function,
                                    'validation_percentage': validation_percentage,
                                    'mape': mape
                                })

                                # Plot scatter plot and error evolution
                                plot_scatter(predictions, y, dataset, mape)
                                plot_error_evolution(nn, dataset)

    return results

def train_neural_network(dataset, input_columns, output_column, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage):
    X = dataset[input_columns].values
    y = dataset[output_column].values.reshape(-1, 1)

    nn = MyNeuralNetwork(num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
    
    nn.fit(X, y)
    
    X_test = X
    predictions = nn.predict(X_test)

    return nn, predictions

# Load the three datasets
dataset1 = pd.read_csv('A1-turbine.csv')
dataset2 = pd.read_csv('A1-synthetic.csv')
dataset3 = pd.read_csv('A1-real_estate.csv')

# Defining input and output columns for each dataset
input_columns1 = ['height_over_sea_level', 'fall', 'net', 'fall_1', 'flow']
output_column1 = 'power_of_hydroelectrical_turbine'

input_columns2 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
output_column2 = 'z'

input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
output_column3 = 'Y house price of unit area'

# Defining neural network configurations for each dataset
num_layers = 4
num_units1 = [len(input_columns1)] + [10, 5, 1]
num_units2 = [len(input_columns2)] + [10, 5, 1]
num_units3 = [len(input_columns3)] + [10, 5, 1]

num_epochs_list = [100, 500, 1000]
learning_rate_list = [0.01, 0.1, 0.2]
momentum_list = [0.0, 0.1, 0.2]
activation_function_list = ['sigmoid', 'relu', 'linear']
validation_percentage_list = [0.2, 0.3, 0.4]

def plot_scatter(predictions, y_true, dataset_name, mape):
    plt.scatter(y_true, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot for {dataset_name} (MAPE: {mape:.2f}%)')
    plt.savefig(f'scatter_plot_{dataset_name}.png')  # Save the plot to a file
    plt.close()

def plot_error_evolution(nn, dataset_name):
    loss_data = nn.loss_epochs()

    plt.plot(loss_data[:, 0], loss_data[:, 1], label='Training Error')
    plt.plot(loss_data[:, 0], loss_data[:, 2], label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Error Evolution for {dataset_name}')
    plt.legend()
    plt.savefig(f'error_evolution_plot_{dataset_name}.png')  # Save the plot to a file
    plt.close()

# Example usage for dataset 1
results1 = explore_parameters(
    dataset=dataset1,
    input_columns=input_columns1,
    output_column=output_column1,
    y=dataset1[output_column1],  # Pass the 'y' values
    num_epochs_list=num_epochs_list,
    learning_rate_list=learning_rate_list,
    momentum_list=momentum_list,
    activation_function_list=activation_function_list,
    validation_percentage_list=validation_percentage_list
)

# Example usage for dataset 2
results2 = explore_parameters(
    dataset=dataset2,
    input_columns=input_columns2,
    output_column=output_column2,
    y=dataset2[output_column2],  # Pass the 'y' values
    num_epochs_list=num_epochs_list,
    learning_rate_list=learning_rate_list,
    momentum_list=momentum_list,
    activation_function_list=activation_function_list,
    validation_percentage_list=validation_percentage_list
)

# Example usage for dataset 3
results3 = explore_parameters(
    dataset=dataset3,
    input_columns=input_columns3,
    output_column=output_column3,
    y=dataset3[output_column3],  # Pass the 'y' values
    num_epochs_list=num_epochs_list,
    learning_rate_list=learning_rate_list,
    momentum_list=momentum_list,
    activation_function_list=activation_function_list,
    validation_percentage_list=validation_percentage_list
)

# ... (rest of the code)


# Create DataFrames for results
results_df1 = pd.DataFrame(results1)
results_df2 = pd.DataFrame(results2)
results_df3 = pd.DataFrame(results3)

# Save results to CSV files
results_df1.to_csv("parameter_results_dataset1.csv", index=False)
results_df2.to_csv("parameter_results_dataset2.csv", index=False)
results_df3.to_csv("parameter_results_dataset3.csv", index=False)
