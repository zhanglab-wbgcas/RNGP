import torch
import pandas as pd
import numpy as np
import time
import random
from model import (same_seeds,train_model,pre_model)
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU ({torch.cuda.get_device_name(0)}) is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available; falling back to CPU.")

same_seeds(2023)

snp_data = pd.read_csv('../wheat2000_geno.csv',header=0)
model_path = '../trained_model.pth'
pheno_data = pd.read_csv('../wheat2000_pheno.csv',header=0)
#output_path_coe = '../coe.txt'
#output_path_mse = '../mse.txt'
snp_data = snp_data.iloc[:,1:]
pheno_data = pheno_data.iloc[:,1]
random_indices = np.random.choice(len(snp_data),500, replace=False)
snp_data = snp_data.iloc[random_indices]
pheno_data = pheno_data.iloc[random_indices]

batch_sizes=32
lr=0.0005

layers = 1
heads = 1
for i in range (10):
    kfold = KFold(n_splits=5, shuffle=True, random_state=i)

    for X_indices, Y_indices in kfold.split(snp_data):
        x_data, x_test = snp_data.iloc[X_indices], snp_data.iloc[Y_indices]
        y_data, y_test = pheno_data.iloc[X_indices], pheno_data.iloc[Y_indices]
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=1/4, random_state=i)
        pca = PCA(n_components=0.95)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)
        x_train=pd.DataFrame(x_train)
        x_val=pd.DataFrame(x_val)
        x_test=pd.DataFrame(x_test)
        features=x_train.shape[1]
        if features %2 == 1:
            d_model=features+1
        else:
            d_model=features
        hidden_size=d_model
        ffn_size=hidden_size*2
        train_model(d_model=d_model,features=features,layers=layers,hidden_size =hidden_size,ffn_size=ffn_size,heads=heads,batch_sizes=batch_sizes,lr=lr,x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,model_path=model_path)
        pre_model(features=features,d_model=d_model,layers=layers,hidden_size =hidden_size,ffn_size=ffn_size,heads=heads,batch_sizes=batch_sizes,x_test=x_test,y_test=y_test,model_path=model_path)


