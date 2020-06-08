# Pouchitis classifiers
Dysbiosis in metabolic genes of the gut microbiomes of patients with inflammatory bowel diseases

This is the code used in the manuscript to generate a machine learning model, i.e. pouchitis classifier, to distinguish between patients with a pouch phenotypes (normal pouch vs. pouchitis) based on bacterial speceis, metabolic pathways or enzmes profiles from shotgun metagenomic data. 
In addition, after training the classifier on the discovery cohort (patients with a normal pouch and with pouchitis, N=208 samples), the classifier can be used to predict which patients with samples defined as recurrent-acute pouchitis phenotypes (from the validation cohort, N=42 samples), will become normal pouch (disease improvement) or pouchitis (disease aggravation) in follow up clinic visits. The prediction performance were: accuracy of ~ 76.2%, sensitivity of 88.9% and specificity of 53.3%.
The classifier is built using the xgboost model, which is an algorithm of gradient boosting trees (GBT). You can change the model and use for example random forest or any other algorithm you prefer, but the code is written to be used specifically with xgboost package. For more information about xgboost, including a nice introduction to boosted trees, go to https://xgboost.readthedocs.io/en/latest/tutorials/model.html

## Python modules requirements
You need to have Python version >=3.0 and the following module installed:
xgboost
sklearn
pandas
numpy
matplotlib
In addition, you need to install XGBoost (eXtreme Gradient Boosting) module. If you are usually installing Python modules with pip, use:
```pip3 install xgboost```
If you are working with Conda, use:
`conda install -c conda-forge xgboost`

## Usage instructions

