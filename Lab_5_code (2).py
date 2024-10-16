import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'unclean-wine-quality.csv'
dataset = pd.read_csv(file_path)

# Question 1

# Drop the first and last columns
dataset = dataset.iloc[:, 1:-1]

# Getting indexes for NaN
nan_indices = np.where(dataset.isnull())

# Count the total number of NaNs
total_nans = dataset.isnull().sum().sum()

# Getting the indexes for NaN
dash_indices = np.where(dataset == '-')

# Count the total number of '-'s
total_dashes = (dataset == '-').sum().sum()

# Replace all '-'s with NaNs
dataset.mask(dataset == '-', other=np.nan, inplace=True)

dataset = dataset.astype('float64')

dataset_q2 = dataset.copy()
dataset_q3 = dataset.copy()
dataset_q4 = dataset.copy()

print(f"The indices of the NaNs are: \n{nan_indices}\n\n The total number of the NaNs inside the dataset is: {total_nans}\n\n")
print(f"The indices of the '-'s are: \n{dash_indices}\n\n The total number of the '-'s inside the dataset is: {total_dashes}\n\n")






# Question 2
constants = {
    'fixed acidity': 0,
    'volatile acidity': 0,
    'citric acid': 0,
    'residual sugar': 0,
    'chlorides': 1,
    'free sulfur dioxide': 0,
    'total sulfur dioxide': 0,
    'density': 0,
    'pH': 1,
    'sulphates': 1,
    'alcohol': 0
}

dataset_q2.fillna(value=constants, inplace=True)

total_nans_after_filling_values = dataset_q2.isna().sum().sum()
print(f"Total Nans after filling with constants: {total_nans_after_filling_values}\n\n")







#Question 3
nan_indices = dataset_q3.isna().to_numpy().nonzero()

for i in range(len(nan_indices[0])):
    row_index = nan_indices[0][i]
    col_index = nan_indices[1][i]
    

    if row_index > 0:
        print(f"Value at index [{row_index-1}, {col_index}] before filling NaN:", dataset_q3.iloc[row_index-1, col_index])
    
    dataset_q3.fillna(method='ffill', inplace=True)
    
    print(f"Value at index [{row_index}, {col_index}] after filling NaN using sample-and-hold:", dataset_q3.iloc[row_index, col_index])
print("\n\n")







#Question 4

# Fill NaN values using linear interpolation
dataset_interpolated = dataset_q4.interpolate(method='linear', axis=0)

# Print the value at [17, 0]
value_at_first_nan = dataset_interpolated.iloc[17, 0]
print(f"The Value at the first index of NaN after performing linear interpolation is {value_at_first_nan}\n")








#Question 5

# Load the noisy signal data
dataset = pd.read_csv("noisy-sine.csv")
noisy_signal = dataset.iloc[:, 0].values

# Window sizes
window_sizes = [5, 31, 51]

# Apply average filter for window size 5
filtered_signal_5 = pd.Series(noisy_signal).rolling(window=5).mean()

# Apply average filter for window size 31
filtered_signal_31 = pd.Series(noisy_signal).rolling(window=31).mean()

# Apply average filter for window size 51
filtered_signal_51 = pd.Series(noisy_signal).rolling(window=51).mean()

# Plot the graph
t = np.arange(len(noisy_signal))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, noisy_signal, label='Original Noisy Signal', alpha=0.5)
ax.plot(t, filtered_signal_5, label='Moving Average (Window Size 5)')
ax.plot(t, filtered_signal_31, label='Moving Average (Window Size 31)')
ax.plot(t, filtered_signal_51, label='Moving Average (Window Size 51)')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.set_title('Original Noisy Signal vs. Moving Average Filtered Signals')
ax.legend()
ax.grid(True)
plt.show()







#Question 6

file_path = "ECG-sample.csv"

ecg_data = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.plot(ecg_data['51392'], color='b', label='ECG Signal')
plt.title('ECG Signal')
plt.grid(True)
plt.legend()
plt.show()

rolling_stats = ecg_data.rolling(window=31).agg(['mean', 'std', 'max', 'min'])

features = rolling_stats.dropna()

print(features)