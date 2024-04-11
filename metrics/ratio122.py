import torch
import pandas as pd
import numpy as np
import time
import random
from model import (same_seeds, train_model, pre_model)
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Check if GPU is available, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU ({torch.cuda.get_device_name(0)}) is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available; falling back to CPU.")

# Function to set random seeds for reproducibility
def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set random seeds
same_seeds(2023)

# Load SNP data
snp_data = pd.read_csv('/input/snp.csv', header=None)
# Define the model path
model_path = '/out/model.pth'
# Load phenotype data
pheno_data = pd.read_csv('/input/phe.csv', header=None)
# Define the output paths for coefficients and MSE
output_path_coe = '/out/coe.txt'
output_path_mse = '/out/mse.txt'

# Model hyperparameters
heads = 32
gap = 1024
batch_sizes = 64
lr = 0.0001
hidden_size = gap
ffn_size = hidden_size * 2

# Start timing model training
start_model = time.time()

# Perform k-fold cross-validation with 20 iterations
# The ratio for training, validation and testing sets is 1:2:2.
for i in range(20):
    kfold = KFold(n_splits=5, shuffle=True, random_state=i)

    # Iterate over each fold
    for X_indices, Y_indices in kfold.split(snp_data):
        x_data, x_train = snp_data.iloc[X_indices], snp_data.iloc[Y_indices]
        y_data, y_train = pheno_data.iloc[X_indices], pheno_data.iloc[Y_indices]
        x_val, x_test, y_val, y_test = train_test_split(x_data, y_data, test_size=1/2, random_state=i)
            
        # Train the model
        train_model(gap=gap, hidden_size=hidden_size, ffn_size=ffn_size, heads=heads, batch_sizes=batch_sizes,
                    lr=lr, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, model_path=model_path)
        
        # Evaluate the model
        pre_model(gap=gap, hidden_size=hidden_size, ffn_size=ffn_size, heads=heads, batch_sizes=batch_sizes,
                  x_test=x_test, y_test=y_test, model_path=model_path, output_path_coe=output_path_coe,
                  output_path_mse=output_path_mse)

# End timing model training
end_model = time.time()

# Compute mean coefficient of determination (COE)
with open(output_path_coe, 'r') as file:
    lines = file.readlines()
    # Extract numerical values from each line
    value1 = [float(line.strip()) for line in lines]
    mean_value1 = sum(value1) / len(value1)

    # Append the mean value to the last line of the text file
    with open(output_path_coe, 'a') as f:
        running_time_seconds = end_model - start_model
        running_time_divided = running_time_seconds / 20
        f.write(f"COE:{mean_value1}\nRunning time : {running_time_divided} Seconds\n")

# Compute mean Mean Squared Error (MSE)
with open(output_path_mse, 'r') as file:
    lines = file.readlines()
    # Extract numerical values from each line
    value2 = [float(line.strip()) for line in lines]
    mean_value2 = sum(value2) / len(value2)

    # Append the mean value to the last line of the text file
    with open(output_path_mse, 'a') as f:
        f.write(f"MSE:{mean_value2}")
