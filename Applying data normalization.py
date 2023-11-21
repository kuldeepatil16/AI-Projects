import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Load the datasets
dataset1 = pd.read_csv('A1-turbine.csv')
dataset2 = pd.read_csv('A1-synthetic.csv')
dataset3 = pd.read_csv('A1-real_estate.csv')

# Create a Min-Max scaler for input and output variables
scaler = MinMaxScaler()

# Define input and output columns for each dataset
# Dataset 1 column names
input_columns1 = ['height_over_sea_level', 'fall', 'net', 'fall_1', 'flow']
output_column1 = 'power_of_hydroelectrical_turbine'

# Dataset 2 column names
input_columns2 = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
output_column2 = 'z'

# Dataset 3 column names
input_columns3 = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
output_column3 = 'Y house price of unit area'

# Impute NaN values with the mean for each dataset
for column in input_columns1:
    dataset1[column] = dataset1[column].fillna(dataset1[column].mean())

for column in input_columns2:
    dataset2[column] = dataset2[column].fillna(dataset2[column].mean())

for column in input_columns3:
    dataset3[column] = dataset3[column].fillna(dataset3[column].mean())

# Normalize input variables in each dataset
dataset1[input_columns1] = scaler.fit_transform(dataset1[input_columns1])
dataset2[input_columns2] = scaler.fit_transform(dataset2[input_columns2])
dataset3[input_columns3] = scaler.fit_transform(dataset3[input_columns3])

# Normalize output variables in each dataset
output_scaler1 = MinMaxScaler()
dataset1[output_column1] = output_scaler1.fit_transform(dataset1[[output_column1]])

output_scaler2 = MinMaxScaler()
dataset2[output_column2] = output_scaler2.fit_transform(dataset2[[output_column2]])

output_scaler3 = MinMaxScaler()
dataset3[output_column3] = output_scaler3.fit_transform(dataset3[[output_column3]])

# Perform outlier detection using Z-score for each dataset
z_score_threshold = 3  # Adjust the threshold as needed

# Z-score for Dataset 1
z_scores1 = np.abs(stats.zscore(dataset1[input_columns1]))
outlier_indices1 = np.where(z_scores1 > z_score_threshold)
outliers_dataset1 = dataset1.iloc[outlier_indices1[0]]

# Z-score for Dataset 2
z_scores2 = np.abs(stats.zscore(dataset2[input_columns2]))
outlier_indices2 = np.where(z_scores2 > z_score_threshold)
outliers_dataset2 = dataset2.iloc[outlier_indices2[0]]

# Z-score for Dataset 3
z_scores3 = np.abs(stats.zscore(dataset3[input_columns3]))
outlier_indices3 = np.where(z_scores3 > z_score_threshold)
outliers_dataset3 = dataset3.iloc[outlier_indices3[0]]

# Display the outlier rows for each dataset
print("Outliers in Dataset 1:")
print(outliers_dataset1)

print("Outliers in Dataset 2:")
print(outliers_dataset2)

print("Outliers in Dataset 3:")
print(outliers_dataset3)

# Save the normalized datasets to separate CSV files
dataset1.to_csv('normalized_dataset1.csv', index=False)
dataset2.to_csv('normalized_dataset2.csv', index=False)
dataset3.to_csv('normalized_dataset3.csv', index=False)