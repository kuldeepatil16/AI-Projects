import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
        errors = []
        for i in range(X.shape[0]):
            self.feed_forward(X[i])
            error = 0.5 * np.sum((self.xi[self.L - 1] - y[i]) ** 2)
            errors.append(error)
        return errors

    def predict(self, X):
        predictions = []
        for sample in X:
            self.feed_forward(sample)
            prediction = self.xi[self.L - 1].copy()
            # Replace NaN values with 0 (you can choose another value if needed)
            prediction[np.isnan(prediction)] = 0
            predictions.append(prediction)
        return np.array(predictions)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_train_samples = X_train.shape[0]
        n_val_samples = X_val.shape[0] if X_val is not None else 0

        for epoch in range(self.num_epochs):
            epoch_errors = []
            for i in range(n_train_samples):
                sample = X_train[i]
                target = y_train[i]

                self.feed_forward(sample)
                self.backpropagate(target)
                self.update_weights()

            train_errors = self.calculate_total_error(X_train, y_train)
            epoch_errors.append(np.mean(train_errors))

            val_errors = None
            if n_val_samples > 0:
                val_errors = self.calculate_total_error(X_val, y_val)
                epoch_errors.append(np.mean(val_errors))
            else:
                epoch_errors.append(0)  # Replace NaN with 0 for validation error
                val_errors = []  # Assign an empty list when there are no validation samples

            self.training_error.append(np.mean(train_errors))
            self.validation_error.append(np.mean(val_errors) if val_errors is not None else 0)

            max_len = min(len(self.training_error), len(self.validation_error), len(epoch_errors))
            epoch_errors += [0] * (max_len - len(epoch_errors))
            self.validation_error += [0] * (max_len - len(self.validation_error))

    def loss_epochs(self):
        return np.column_stack((self.training_error, self.validation_error))

    def preprocess_dataset(self, dataset, input_columns, output_column, validation_percentage, random_state):
        # Rearrange columns to have input features first and output variable last
        dataset = dataset[[*input_columns, output_column]]

       # Drop rows with NaN values
        dataset = dataset.dropna()


       # Create a Min-Max scaler for input and output variables
        scaler = MinMaxScaler()
       # Normalize input and output variables
        dataset[input_columns] = scaler.fit_transform(dataset[input_columns])
        dataset[output_column] = scaler.fit_transform(dataset[[output_column]])

       # Shuffle the data
        shuffled_dataset = dataset.sample(frac=1, random_state=random_state)

       # Set the test_size variable based on the dataset
        if "A1-turbine" in dataset.columns:
          # File: A1-turbine.txt
          test_size = 0.15
        elif "A1-synthetic" in dataset.columns:
           # File: A1-synthetic.txt
           test_size = 0.2
        elif "A1-real_estate" in dataset.columns:
           # A1-real_estate
           # Select randomly 80% of the patterns for training and validation
           test_size = 0.2
        else:
           # Handle other datasets if needed
           test_size = 0.2

          # Check if the validation percentage is 0
        if validation_percentage == 0:
          # Use all data for training, no validation set
          train_set = shuffled_dataset
          val_set = pd.DataFrame(columns=dataset.columns)  # Empty DataFrame for consistency
        else:
          # Split the dataset into training, validation, and test sets
          train_val_set, test_set = train_test_split(shuffled_dataset, test_size=test_size, random_state=random_state)

        # Further split training/validation set into training and validation sets
          train_size = (1 - test_size) * (1 - validation_percentage)
          train_set, val_set = train_test_split(train_val_set, test_size=train_size, random_state=random_state)

        return train_set, val_set, test_set

    def train_neural_network(self, dataset, input_columns, output_column, num_layers, num_units, num_epochs, learning_rate, momentum, activation_function, validation_percentage):
        # Preprocess dataset
        X = dataset[input_columns].values
        y = dataset[output_column].values.reshape(-1, 1)

        train_set, val_set, _ = self.preprocess_dataset(dataset, input_columns, output_column, validation_percentage, random_state=42)

        X_train = train_set[input_columns].values
        y_train = train_set[output_column].values.reshape(-1, 1)

        X_val = val_set[input_columns].values
        y_val = val_set[output_column].values.reshape(-1, 1)

        self.fit(X_train, y_train, X_val)

        X_test = X
        predictions = self.predict(X_test)

        return predictions
 
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

num_epochs = 1000
learning_rate = 0.1
momentum = 0.0
activation_function = "sigmoid"
validation_percentage = 0.2

# Training the neural network for each dataset
nn1 = MyNeuralNetwork(num_layers, num_units1, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
predictions1 = nn1.train_neural_network(dataset1, input_columns1, output_column1, num_layers, num_units1, num_epochs, learning_rate, momentum, activation_function, validation_percentage)

nn2 = MyNeuralNetwork(num_layers, num_units2, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
predictions2 = nn2.train_neural_network(dataset2, input_columns2, output_column2, num_layers, num_units2, num_epochs, learning_rate, momentum, activation_function, validation_percentage)

nn3 = MyNeuralNetwork(num_layers, num_units3, num_epochs, learning_rate, momentum, activation_function, validation_percentage)
predictions3 = nn3.train_neural_network(dataset3, input_columns3, output_column3, num_layers, num_units3, num_epochs, learning_rate, momentum, activation_function, validation_percentage)

# Saving predictions to files
# np.savetxt("predictions_dataset1.csv", predictions1, delimiter=",")
np.savetxt("predictions_dataset2.csv", predictions2, delimiter=",")
np.savetxt("predictions_dataset3.csv", predictions3, delimiter=",")

# Saving loss data to files
loss_data1 = nn1.loss_epochs()
np.savetxt("loss_data_dataset1.csv", loss_data1, delimiter=",")

loss_data2 = nn2.loss_epochs()
np.savetxt("loss_data_dataset2.csv", loss_data2, delimiter=",")

loss_data3 = nn3.loss_epochs()
np.savetxt("loss_data_dataset3.csv", loss_data3, delimiter=",")
