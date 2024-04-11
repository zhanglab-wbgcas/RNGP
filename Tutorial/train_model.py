import torch
import pandas as pd
import argparse
from model import (same_seeds, train_model)
from sklearn.model_selection import train_test_split

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model with given data.")
    parser.add_argument("--train_snp", type=str, help="Path to training SNP data")
    parser.add_argument("--train_phe", type=str, help="Path to training phenotype data")
    parser.add_argument("--model", type=str, help="Path to save trained model")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU ({torch.cuda.get_device_name(0)}) is available and will be used.")
    else:
        device = torch.device("cpu")
        print("GPU is not available; falling back to CPU.")

    # Set random seeds
    same_seeds(2023)

    # Load training data
    train_snp = pd.read_csv(args.train_snp, header=None)
    train_phe = pd.read_csv(args.train_phe, header=None)
    model_path = args.model + '/model.ph'

    # Split dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(train_snp, train_phe, test_size=1/9, random_state=2023)
    
    # Model hyperparameters
    heads = 32
    gap = 1024
    batch_sizes = 64
    lr = 0.0001
    hidden_size = gap
    ffn_size = hidden_size * 2

    # Train the model
    train_model(gap=gap, hidden_size=hidden_size, ffn_size=ffn_size, heads=heads,
                batch_sizes=batch_sizes, lr=lr, x_train=x_train, y_train=y_train,
                x_val=x_val, y_val=y_val, model_path=model_path)
