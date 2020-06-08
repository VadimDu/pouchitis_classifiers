#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun Jun  7 14:33:03 2020

#Classifier to distinguish between patients with a pouch phenotypes (normal pouch vs. pouchitis) based on bacterial speceis, metabolic pathways or enzmes profiles.
#Author: Vadim (Dani) Dubinsky (dani.dubinsky@gmail.com)

"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, plot_confusion_matrix

try:
    import xgboost as xgb ##XGBoost, extreme gradient boosting model
except:
    sys.exit("This program requires Python3 xgboost module, please install it and try again")


#Load the data with the samples from the validation cohort (with recurrent-acute pouchitis) and its labels.
#The classifier can be used to predict which patients with samples defined as recurrent-acute pouchitis phenotypes (from the validation cohort, N=42 samples)

target_labels_ra = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/metadata_n42_RMC_Pouch_cohort_for_ML_RA_last_followup_pheno.txt", index_col=0, sep="\t")
taxa_data_ra = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/Taxa_species_profile_cpm_n42_RMC_Pouch_cohort_RA_validation.txt", index_col=0, sep="\t")

#Create a new table with only a subset of the original features, based on empirical number of highest scoring features (user selected) 
#We need to use the same selected features both in the training and validation sets
taxa_data_ra_sub = subset_features_in_df(features_scores_tree_based.index[0:60], taxa_data_ra)
#The highest scoring 40 features (species) obtained in the original manuscript:
taxa_data_ra_sub = subset_features_in_df(best_features_species_40.index, taxa_data_ra)

#Add fecal calprotectin as a preditcor (i.e. feature)
taxa_data_ra_sub["calpro"] = target_labels_ra.Calprotectin_log
taxa_data_sub["calpro"] = target_labels.Calprotectin_log

### Test the model on the [unseen by the model] validation set to predict the future phenotype (normal pouch or pouchitis) of samples defined as "recurrent-acute pouchitis": ###
sensitivity, specificity, accuracy, AUC = [],[],[],[]
#Train the model ("fit") on the training dataset
model.fit(taxa_data_sub, target_labels["labels_binary"])
#Use the model to predict the label of the validation dataset samples
y_val_pred = model.predict(taxa_data_ra_sub)
accuracy.append(accuracy_score(target_labels_ra["labels_binary"], y_val_pred) * 100.0)
tn, fp, fn, tp = confusion_matrix(target_labels_ra["labels_binary"], y_val_pred).ravel()
sensitivity.append(tp/(tp+fn))
specificity.append(tn/(tn+fp))
y_val_pred_proba = model.predict_proba(taxa_data_ra_sub) #extract the probabilities of prediction, to calculate AUC and average prediction
fpr1, tpr1, thresholds = roc_curve(target_labels_ra["labels_binary"], y_val_pred_proba[:,1]) #To calculate the ROC curve we need the true positive rate (TPR) and the false positive rate (FPR). We give as input to the func. our True Positive probabilities of predictions (probs[:,1]) and the true target labels (y_test).
AUC.append(auc(fpr1, tpr1))

#Summarize the testing results:
print("Mean sensitivity: %.3f (%.3f); and specificity: %.3f (%.3f)" %
     (np.mean(sensitivity), np.std(sensitivity), np.mean(specificity), np.std(specificity)))
print("Mean accuracy: %.2f%% (%.2f)" % (np.mean(accuracy), np.std(accuracy)))
print("Mean AUC: %.3f (%.3f)" % (np.mean(AUC), np.mean(AUC)))

#Plot confusion matrix (tn, fp, fn, tp) the visualise the number of correctly and incorrectly classified (predicted) samples:
plt.rcParams["font.weight"] = "bold" #"normal"
plt.rcParams["axes.labelweight"] = "bold" #"normal"
plt.figure(figsize=(10,10))
plot_confusion_matrix(model, taxa_data_ra_sub, target_labels_ra["labels_binary"])

#Plot the AUC ROC:
plot_AUC_ROC (fpr1, tpr1)


#----------------------Utility functions:-----------------------------------###

def subset_features_in_df(features_list, df):
    '''Returns a subsetted dataframe including only the specified samples or features
       Assuming the features (e.g species/genes) are in the columns and samples are in rows'''
    subset_df = pd.DataFrame()
    missing = []
    for taxa in features_list:
        if (taxa in df.columns): #first check if each feature exists in the df, only then subset these features
            subset_df[taxa] = df.loc[:, taxa]
        else:
            missing.append(taxa)
    print("Missing feaures/samples:", missing)
    return (subset_df)

def plot_AUC_ROC (fpr, tpr):
    '''Function to plot ROC AUC curve. Inputs are the false positive rate (fpr) and true positive rate (tpr) that you obtained from roc_curve()
       Labels and colors need to be changed manually'''
    fig = plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkred', lw=3, label='XGboost species, top k=40 + calpro (AUC = %0.3f)' % auc(fpr, tpr))
    plt.plot([0,1], [0,1], color='darkblue', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim([-0.01,1])
    plt.ylim([-0.01,1.01])
    plt.legend(loc="lower right", fontsize=14)
    plt.title("Model for Recurrent-Acute patients Pouchitis classification (species-based)", fontsize=18, pad=10)
    
###-------------------------------------------------------------------------###
