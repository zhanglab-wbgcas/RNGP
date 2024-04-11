# plantGPT
># Introduction

plantGPT(a plant genomic prediction model based on Transformer), a Transformer-based model, can be used to predict phenotypes using genomic data.

Instructions and examples are provided in the following tutorials.
># Requirement
```
Python 3.9.18
PyTorch >= 1.5.0
numpy
pandas
sklearn
random
math
tqdm
```
## Input file
```
train_model.py:
--train_snp Path to training SNP data
--train_phe Path to training phenotype data
--model Path to save trained model 
predict.py:
--test_snp Path to test SNP data
--test_phe Path to control phenotype data
--model Path to saved trained model
```
example:
python train_model.py --train_snp /train_snp.csv --train_phe /train_phe.csv --model /out
python predict.py --test_snp /test_snp.csv --test_phe /test_phe.csv --model /out/model.ph --out /out
```
## Output file
```
After the plantGPT train_model.py, the model will be save at: "out/model.pth".
After the predict.py, coe.txt, mse.txt and preds.txt will be save at: "/out/".
coe.txt: Pearson coefficients for predicted and true values.
mse.txt: Mean square error of predicted and true values.
preds.txt: Predicted value.

```

[//]: # (```)


># Paper Link
