import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import (DataLoader,Dataset)
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler 
from torch.utils.data import  DataLoader
from retnet import RetNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Check for GPU availability and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU ({torch.cuda.get_device_name(0)}) is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available; falling back to CPU.")
# Function to set the same seed for reproducibility
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
# EarlyStopping class to monitor validation score and stop training if no improvement
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.Inf  
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
# Function to prepare features and labels from SNP data and phenotype
def getXY(features,snp, phenotype):

    snp_list = []
    for i in range(len(snp)):
        sample = snp.iloc[i, :]

        feature = []
        or_length = len(sample)
        if features %2 == 1:
            features = features + 1
        padded_sample = np.pad(sample, (0, features - or_length if or_length < features else 0), mode='constant')
        
        feature.append(padded_sample)
        feature = np.asarray(feature, dtype=np.float32)

        snp_list.append(feature)

    snp_list = np.asarray(snp_list) #(n_samples, 1, features)
    
    y_train = phenotype.values  
    y_train = y_train.astype(np.float32)
    y_train = pd.DataFrame(y_train)

    return snp_list, y_train

# Custom Dataset class to handle data loading
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
    
# PlantGPT class that defines the model architecture
class plantGPT(nn.Module):
    def __init__(self, input_dim,layers,hidden_size,ffn_size,nhead=4, d_model=64):
        super().__init__()
        self.retnet = RetNet(layers, hidden_size, ffn_size, heads=nhead, double_v_dim=True)
        if d_model>200:
            self.pred = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model//2),
                nn.GELU(),
                nn.Linear(d_model//2,d_model//4),
                nn.GELU(),
                nn.Linear(d_model//4, 1),
            )
        else:
            self.pred = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.Linear(128, 1)
            )
    def forward(self, mels):
        x = mels.permute(1, 0, 2)
        x = self.retnet(x)
        x = x.transpose(0, 1)
        x = x.squeeze(1)
        x = self.pred(x)
        return x
    
# Function to train the model with given parameters
def train_model(features,d_model,layers, hidden_size, ffn_size,heads, batch_sizes, x_train, y_train, x_val, y_val, lr, model_path):
    """
    This function trains the plantGPT model with the provided parameters and data.
    It includes early stopping and learning rate scheduling.
    """
    """
    Parameters:
    - windows: Window size for segmenting the SNP data.
    - d_model: Dimension of the model.
    - layers: Number of layers in the RetNet.
    - hidden_size: Hidden size in each layer of the RetNet.
    - ffn_size: Feedforward network size in each layer of the RetNet.
    - heads: Number of attention heads in the RetNet.
    - batch_sizes: Batch size for training and validation.
    - x_train: Training features.
    - y_train: Training labels.
    - x_val: Validation features.
    - y_val: Validation labels.
    - lr: Learning rate for the optimizer.
    - model_path: Path to save the best model.
    - n_epochs: the times of Training

    """
    d_model = d_model
    n_epochs = 25

    x_train, y_train = getXY(features, x_train, y_train)
    x_val, y_val = getXY(features, x_val, y_val)

    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
    
    model = plantGPT(input_dim=d_model,layers=layers,hidden_size =hidden_size,ffn_size=ffn_size,nhead=heads, d_model=d_model)
    model.to(device)

    # Setting loss function
    criterion = nn.MSELoss()
    # Setting optimization function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Creating the ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Setting the training dataset
    train_dataset = DataSet(data=x_train, label=y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)
    
    # Setting the test dataset
    val_dataset = DataSet(data=x_val, label=y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)


    best_coe = -float('inf') 
    best_model_state = None
    y_val = y_val.iloc[:, 0]  
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
        with torch.no_grad():  
            for batch in tqdm(val_loader):
                data, _ = batch
                data = data.to(device)
                batch_preds = model(data)
                preds.extend(batch_preds.cpu().numpy())
        preds = np.concatenate(preds, axis=0) 
        mse = mean_squared_error(y_val, preds)
        coe = np.corrcoef(y_val, preds)[0, 1]
        mae = mean_absolute_error(y_val, preds)
        print(f"Val mse = {mse:.4f} ")
        print(f"Val mae = {mae:.4f} ")
        print(f"Val coe = {coe:.4f} ")
        if coe > best_coe:
            best_coe = coe
            best_model_state = model.state_dict().copy() 
        early_stopping.early_stopping(coe)  

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

# Function to predict using the trained model

def pre_model(features,d_model,layers, hidden_size, ffn_size,heads,batch_sizes,x_test,y_test,model_path,output_path_coe=None,output_path_mse=None):
    """
    This function predicts using the trained plantGPT model and evaluates its performance
    on the test data. It can also save the results to the specified output paths.
    """
    """
    Parameters:
    - x_test: Test features.
    - y_test: Test labels.
    - model_path: Path to load the trained model.
    - output_path_coe: Path to save the coefficient results, default=None.
    - output_path_mse: Path to save the MSE results, default=None.
    """

    d_model = d_model
    x_test,y_test=getXY(features,x_test,y_test)
    
    model = plantGPT(input_dim=d_model, layers=layers,hidden_size =hidden_size,ffn_size=ffn_size,nhead=heads, d_model=d_model)
    model.to(device)
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict['model'])

    test_dataset = DataSet(data=x_test,label =y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False,
                                pin_memory=True)

    model.eval()

    preds = []
    with torch.no_grad():  
        for batch in tqdm(test_loader):
            data, _ = batch
            data = data.to(device)
            batch_preds = model(data)
            preds.extend(batch_preds.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    y_test = y_test.iloc[:, 0] 
    coe = np.corrcoef(y_test, preds)[0, 1]
    mse = mean_squared_error(y_test, preds)
    print(f"Test coe = {coe:.6f} ")
    print(f"Test mse = {mse:.6f} ")
    #with open(output_path_coe, 'a') as f:
        #f.write(f"{coe}\n")
    #with open(output_path_mse, 'a') as f:
        #f.write(f"{mse}\n") 
