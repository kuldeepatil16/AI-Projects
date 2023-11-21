import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Function to preprocess a dataset
def preprocess_dataset(dataset, input_columns, output_column, test_size, random_state):
    # Rearrange columns to have input features first and output variable last
    dataset = dataset[input_columns + [output_column]]

    # Create a Min-Max scaler for input and output variables
    scaler = MinMaxScaler()

    # Normalize input and output variables
    dataset[input_columns] = scaler.fit_transform(dataset[input_columns])
    dataset[output_column] = scaler.fit_transform(dataset[[output_column]])

    # Shuffle the data and split it into training/validation and test sets
    shuffled_dataset = dataset.sample(frac=1, random_state=random_state)
    train_val_set, test_set = train_test_split(shuffled_dataset, test_size=test_size, random_state=random_state)

    # Further split training/validation set into training and validation sets
    train_set, val_set = train_test_split(train_val_set, test_size=0.5, random_state=random_state)

    return train_set, val_set, test_set

# Load the datasets
dataset1 = pd.read_csv('A1-turbine.csv')
dataset2 = pd.read_csv('A1-synthetic.csv')
dataset3 = pd.read_csv('A1-real_estate.csv')

# Dataset 1 preprocessing
input_columns1 = ['height_over_sea_level', 'fall', 'net', 'fall_1']
output_column1 = 'power_of_hydroelectrical_turbine'
test_size1 = 0.15
random_state1 = 42

train_set1, val_set1, test_set1 = preprocess_dataset(dataset1, input_columns1, output_column1, test_size1, random_state1)

# Dataset 2 preprocessing
input_columns2 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
output_column2 = 'z'
test_size2 = 0.2
random_state2 = 42

train_set2, val_set2, test_set2 = preprocess_dataset(dataset2, input_columns2, output_column2, test_size2, random_state2)

# Dataset 3 preprocessing
input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
output_column3 = 'Y house price of unit area'
test_size3 = 0.2
random_state3 = 42

train_set3, val_set3, test_set3 = preprocess_dataset(dataset3, input_columns3, output_column3, test_size3, random_state3)

# Display information about the sets for each dataset
print("Dataset 1:")
print("Training set shape:", train_set1.shape)
print("Validation set shape:", val_set1.shape)
print("Test set shape:", test_set1.shape)

print("\nDataset 2:")
print("Training set shape:", train_set2.shape)
print("Validation set shape:", val_set2.shape)
print("Test set shape:", test_set2.shape)

print("\nDataset 3:")
print("Training set shape:", train_set3.shape)
print("Validation set shape:", val_set3.shape)
print("Test set shape:", test_set3.shape)
