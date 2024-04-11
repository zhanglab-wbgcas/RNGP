import torch
import pandas as pd
import argparse
from model import (same_seeds, pre_model)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model with given data.")
    parser.add_argument("--test_snp", type=str, help="Path to test SNP data")
    parser.add_argument("--test_phe", type=str, help="Path to test phenotype data")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--out", type=str, help="Path to output directory")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU ({torch.cuda.get_device_name(0)}) is available and will be used.")
    else:
        device = torch.device("cpu")
        print("GPU is not available; falling back to CPU.")

    # Load test data
    x_test = pd.read_csv(args.test_snp, header=None)
    y_test = pd.read_csv(args.test_phe, header=None)
    model_path = args.model
    # Output file paths
    output_path_coe = args.out + '/coe.txt'
    output_path_mse = args.out + '/mse.txt'
    output_path_preds = args.out + '/preds.txt'
    # Model hyperparameters
    heads = 32
    gap = 1024
    batch_sizes = 64
    hidden_size = gap
    ffn_size = hidden_size * 2

    # Predict using pre-trained model
    pre_model(gap=gap, hidden_size=hidden_size, ffn_size=ffn_size, heads=heads,
              batch_sizes=batch_sizes, x_test=x_test, y_test=y_test, model_path=model_path,
              output_path_coe=output_path_coe, output_path_mse=output_path_mse,
              output_path_preds=output_path_preds)
