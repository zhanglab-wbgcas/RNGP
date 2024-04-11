import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings
import math
import pandas as pd
from tqdm.auto import tqdm  # Progress bar
from torch.utils.data import (DataLoader, Dataset)
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler  # Import learning rate scheduler
from torch.utils.data import DataLoader
from retnet import RetNet  # Import RetNet module
from sklearn.metrics import mean_squared_error

if torch.cuda.is_available():
    device = torch.device("cuda")  # Check if CUDA is available
    print(f"GPU ({torch.cuda.get_device_name(0)}) is available and will be used.")
else:
    device = torch.device("cpu")  # Fall back to CPU if CUDA is not available
    print("GPU is not available; falling back to CPU.")

def same_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value.
    """
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

class EarlyStopping:
    """
    Early stopping to terminate training when validation score doesn't improve.

    Args:
        patience (int): Number of epochs to wait before early stopping.
        verbose (bool): If True, prints a message for each epoch when early stopping is triggered.
        delta (float): Minimum change in the monitored quantity to qualify as improvement.
    """
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.Inf  # Initialize as a negative number
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def early_stopping(self, val_score):
        """
        Check if early stopping criteria are met.

        Args:
            val_score (float): Current validation score.
        """
        if val_score > self.best_score + self.delta:
            self.counter = 0
            self.best_score = val_score
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

def getXY(gap, snp, phenotype):
    """
    Convert data into sub-vectors.

    Args:
        gap (int): Gap size.
        snp (DataFrame): Input SNP data.
        phenotype (DataFrame): Target phenotype data.

    Returns:
        snp_list (ndarray): Array of SNP sub-vectors.
        y_train (DataFrame): Target phenotype data.
    """
    snp_list = []
    for i in range(len(snp)):
        sample = snp.iloc[i, :]
        feature = []
        length = len(sample)
        # Split the vector into sub-vectors of length 'gap'
        for k in range(0, length, gap):
            if (k + gap <= length):
                a = sample[k:k + gap]
            else:
                a = sample[length - gap:length]
            feature.append(a)
        feature = np.asarray(feature, dtype=np.float32)
        snp_list.append(feature)

    snp_list = np.asarray(snp_list)  # (n_samples, gap_num, gap)

    y_train = phenotype.values  
    y_train = y_train.astype(np.float32)
    y_train = pd.DataFrame(y_train)

    return snp_list, y_train

class DataSet(Dataset):
    """
    Custom dataset class for loading data.

    Args:
        data (ndarray): Input data.
        label (DataFrame): Target labels.
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label.iloc[index], dtype=torch.float32)
        return data, label

class plantGPT(nn.Module):
    """
    Custom plantGPT model.

    Args:
        input_dim (int): Input dimension.
        hidden_size (int): Hidden size for the model.
        ffn_size (int): Size of the feedforward neural network.
        nhead (int): Number of attention heads.
        d_model (int): Dimension of the model.
    """
    def __init__(self, input_dim, hidden_size, ffn_size, nhead=4, d_model=64):
        super().__init__()
        self.retnet = RetNet(hidden_size, ffn_size, heads=nhead, double_v_dim=True)
        self.pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4),
            nn.Linear(d_model//4, d_model//8),
            nn.GELU(),
            nn.Linear(d_model//8, 1)
        )
    def forward(self, mels):
        x = mels.permute(1, 0, 2)
        x = self.retnet(x)
        x = x.transpose(0, 1)
        # Mean pooling
        x = x.mean(dim=1)
        x = self.pred(x)
        return x
def train_model(gap, hidden_size, ffn_size, heads, batch_sizes, x_train, y_train, x_val, y_val, lr, model_path):
    """
    Train the plantGPT model.

    Args:
        gap (int): Gap size.
        hidden_size (int): Hidden size for the model.
        ffn_size (int): Size of the feedforward neural network.
        heads (int): Number of attention heads.
        batch_sizes (int): Batch size.
        x_train (DataFrame): Training SNP data.
        y_train (DataFrame): Training phenotype data.
        x_val (DataFrame): Validation SNP data.
        y_val (DataFrame): Validation phenotype data.
        lr (float): Learning rate.
        model_path (str): Path to save the trained model.
    """
    gap = gap
    d_models = gap
    n_epochs = 25
    x_train, y_train = getXY(gap, x_train, y_train)
    x_val, y_val = getXY(gap, x_val, y_val)

    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
    
    model = plantGPT(input_dim=d_models, hidden_size=hidden_size, ffn_size=ffn_size, nhead=heads, d_model=d_models)
    model.to(device)

    # Setting loss function
    criterion = nn.MSELoss()
    # Setting optimization function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Creating the ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, verbose=True)

    # Setting the training dataset
    train_dataset = DataSet(data=x_train, label=y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)
    
    # Setting the test dataset
    val_dataset = DataSet(data=x_val, label=y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

    best_coe = -float('inf') 
    best_model_state = None
    y_val = y_val.iloc[:, 0]  # Extracting values from '0' column and converting to 1D array
    for epoch in range(n_epochs):
       
        model.train()
        train_loss = []
        
        for batch in tqdm(train_loader):
            data, target = batch
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        train_loss = sum(train_loss) / len(train_loss)
            
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")
        
        model.eval()
        
        preds = []
        with torch.no_grad():  # Disable gradient calculation
            for batch in tqdm(val_loader):
                data, _ = batch
                data = data.to(device)
                batch_preds = model(data)
                preds.extend(batch_preds.cpu().numpy())
        preds = np.concatenate(preds, axis=0) 
          
        mse = mean_squared_error(y_val, preds)
        coe = np.corrcoef(y_val, preds)[0, 1]

        print(f"Val mse = {mse:.4f} ")
        print(f"Val coe = {coe:.4f} ")

        if coe > best_coe:
            best_coe = coe
            best_model_state = model.state_dict().copy() 
        early_stopping.early_stopping(coe)  # Pass validation set COE score

        scheduler.step(coe)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print("-----------------------------end test-----------------")
    print(f"Val best coe = {best_coe:.4f}")
    if best_model_state is not None:
        save_dict = {
            'model': best_model_state,
        }
        torch.save(save_dict, model_path)

def pre_model(gap, hidden_size, ffn_size, heads, batch_sizes, x_test, y_test, model_path, output_path_coe, output_path_mse, output_path_preds):
    """
    Predict using the trained model.

    Args:
        gap (int): Gap size.
        hidden_size (int): Hidden size for the model.
        ffn_size (int): Size of the feedforward neural network.
        heads (int): Number of attention heads.
        batch_sizes (int): Batch size.
        x_test (DataFrame): Test SNP data.
        y_test (DataFrame): Test phenotype data.
        model_path (str): Path to the trained model.
        output_path_coe (str): Output path for correlation coefficient results.
        output_path_mse (str): Output path for mean squared error results.
        output_path_preds (str): Output path for predictions.
    """
    d_models = gap
               
    x_test, y_test = getXY(gap, x_test, y_test)
    
    model = plantGPT(input_dim=d_models, hidden_size=hidden_size, ffn_size=ffn_size, nhead=heads, d_model=d_models)
    model.to(device)
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict['model'])

    test_dataset = DataSet(data=x_test, label=y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

    model.eval()

    preds = []
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(test_loader):
            data, _ = batch
            data = data.to(device)
            batch_preds = model(data)
            preds.extend(batch_preds.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    y_test = y_test.iloc[:, 0]  # Extract column values and convert to 1D array
    coe = np.corrcoef(y_test, preds)[0, 1]
    mse = mean_squared_error(y_test, preds)
     
    with open(output_path_coe, 'a') as f:
        f.write(f"{coe}\n")
    with open(output_path_mse, 'a') as f:
        f.write(f"{mse}\n") 
    preds = preds.tolist()
    with open(output_path_preds, 'a') as f:
        for pred in preds:
            f.write(f"{pred}\n")
