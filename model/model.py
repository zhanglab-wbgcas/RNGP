import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings
import math
import pandas as pd
from tqdm.auto import tqdm  # tqdm for progress bars
from torch.utils.data import DataLoader, Dataset  # DataLoader and Dataset for handling data
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler  # Learning rate scheduler
from torch.utils.data import DataLoader  # DataLoader for loading data
from retnet import RetNet  # Import RetNet module
from sklearn.metrics import mean_squared_error  # Import mean squared error metric

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set random seeds
same_seeds(2023)

# Custom EarlyStopping class to handle early stopping during training
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.Inf  # Initialize with a negative value
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def early_stopping(self, val_score):
        if val_score > self.best_score + self.delta:
            self.counter = 0
            self.best_score = val_score
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

# Function to split data into sub-vectors and convert to PyTorch tensors
def getXY(gap, snp, phenotype):
    snp_list = []
    for i in range(len(snp)):
        sample = snp.iloc[i, :]
        feature = []
        length = len(sample)
        # Splitting the vector into sub-vectors of length 'gap'
        for k in range(0, length, gap):
            if (k + gap <= length):
                a = sample[k:k + gap]
            else:
                a = sample[length - gap:length]
            feature.append(a)
        feature = np.asarray(feature, dtype=np.float32)
        snp_list.append(feature)
    snp_list = np.asarray(snp_list)
    
    y_train = phenotype.values  
    y_train = y_train.astype(np.float32)
    y_train = pd.DataFrame(y_train)
    return snp_list, y_train

# Custom dataset class for handling data in DataLoader
class DataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label.iloc[index], dtype=torch.float32)
        return data, label

# Custom plantGPT model class inheriting from nn.Module
class plantGPT(nn.Module):
    def __init__(self, input_dim, hidden_size, ffn_size, nhead=4, d_model=64):
        super().__init__()
        # Initialize RetNet module
        self.retnet = RetNet(hidden_size, ffn_size, heads=nhead, double_v_dim=True)
        # Define prediction layers
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
        # Permute input tensor
        x = mels.permute(1, 0, 2)
        # Pass through RetNet module
        x = self.retnet(x)
        x = x.transpose(0, 1)
        # Mean pooling
        x = x.mean(dim=1)
        # Pass through prediction layers
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
        batch_sizes (int): Batch size for training.
        x_train (DataFrame): Training input data.
        y_train (DataFrame): Training target data.
        x_val (DataFrame): Validation input data.
        y_val (DataFrame): Validation target data.
        lr (float): Learning rate.
        model_path (str): Path to save the trained model.
    """
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
    y_val = y_val.iloc[:, 0]  # Extracting '0' column values and converting to a one-dimensional array
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
        early_stopping.early_stopping(coe)  # Pass validation set's coe score

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

def pre_model(gap, hidden_size, ffn_size, heads, batch_sizes, x_test, y_test, model_path, output_path_coe, output_path_mse):
    """
    Predict using the trained plantGPT model.

    Args:
        gap (int): Gap size.
        hidden_size (int): Hidden size for the model.
        ffn_size (int): Size of the feedforward neural network.
        heads (int): Number of attention heads.
        batch_sizes (int): Batch size for prediction.
        x_test (DataFrame): Test input data.
        y_test (DataFrame): Test target data.
        model_path (str): Path to the trained model.
        output_path_coe (str): Path to save the coefficient of correlation (coe).
        output_path_mse (str): Path to save the mean squared error (mse).
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
    y_test = y_test.iloc[:, 0]  # Extract column values and convert to a one-dimensional array
    coe = np.corrcoef(y_test, preds)[0, 1]
    mse = mean_squared_error(y_test, preds)
     
    with open(output_path_coe, 'a') as f:
        f.write(f"{coe}\n")
    with open(output_path_mse, 'a') as f:
        f.write(f"{mse}\n") 
