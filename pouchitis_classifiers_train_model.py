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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

try:
    import xgboost as xgb ##XGBoost, extreme gradient boosting model
except:
    sys.exit("This program requires Python3 xgboost module, please install it and try again")


###----Main code for generating an xgboost model trained on pouch discovery cohort----###

#Load the features data (species / pathways / enzymes) and the labels of the data (metadata) for classification
#Replace the path and name of your file:
target_labels = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/metadata_labels_n208_RMC_Pouch_cohort_for_ML.txt", index_col=0, sep="\t")
taxa_data = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/Taxa_species_profile_cpm_n208_RMC_Pouch_cohort.txt", index_col=0, sep="\t")
metacyc = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/MetaCyc_pathways_profile_cpm_n208_RMC_Pouch_cohort.txt", index_col=0, sep="\t")
enzymes = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/EC4_enzymes_profile_cpm_n208_RMC_Pouch_cohort_0.05filt.txt", index_col=0, sep="\t")
calpro_only = pd.DataFrame(pd.DataFrame({"Calprotectin_log": target_labels["Calprotectin_log"]}, index=target_labels.index))

# XGBoost model setup:
model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree=0.5, subsample=0.5,
                          learning_rate=0.025, reg_lambda=0.01, reg_alpha=0.001, max_depth=3,
                          n_estimators=500, random_state=9, min_child_weight=1)

#K-fold cross validation with imbalance classes, K=5
kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

###Repeat 100 times the K-fold cross-validation, calculate performance matrices for each 5-fold CV repeat and also average feature importance scores:
mean_list, std_list, sensitivity, specificity, acc_list, rp_mean_feat_import = [],[],[],[],[],[]
for _ in range(100):
    results = cross_val_score(model, taxa_data, target_labels["labels_binary"], cv=kfold, scoring='roc_auc', n_jobs=2) #returns AUC score for each k-fold CV
    print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
    mean_list.append(results.mean())
    std_list.append(results.std())
    y_pred_cv = cross_val_predict(model, taxa_data, target_labels["labels_binary"], cv=kfold, method='predict', n_jobs=2) #returns the predicted y (labels) values
    tn, fp, fn, tp = confusion_matrix(target_labels["labels_binary"], y_pred_cv).ravel() #obtain the number of true positives, true negatives, false positives and false negatives
    sensitivity.append(tp/(tp+fn))
    specificity.append(tn/(tn+fp))
    acc_list.append(accuracy_score(target_labels["labels_binary"], y_pred_cv))
    
    ###Obtain model feature importance for each fold within the K-fold CV and repeat to calculte the average importance score:
    cv_model_feat_import = []
    cv_estimator = cross_validate(model, taxa_data, target_labels["labels_binary"], cv=kfold, return_estimator=True)
    for estimator in cv_estimator['estimator']: #cv_estimator is a dict the size of CV fold (k=5)
        cv_model_feat_import.append(estimator.feature_importances_)
    rp_mean_feat_import.append(np.mean(cv_model_feat_import, axis=0)) #mean importance score for each feature across 5-fold CV

#Summarize results of the 100 repeats of 5-fold CV:
print("Mean sensitivity: %.3f (%.3f); and specificity: %.3f (%.3f)" %
     (np.mean(sensitivity), np.std(sensitivity), np.mean(specificity), np.std(specificity)))
print("Mean AUC: %.3f (%.3f)" % (np.mean(mean_list), np.mean(std_list)))   
print("Mean accuracy: %.2f%% (%.2f)" % (np.mean(acc_list), np.std(acc_list))) 

#Save the importance scores of each feature into a dataframe
features_scores_tree_based = pd.DataFrame({"XGboost_average_feature_importance":np.mean(rp_mean_feat_import, axis=0)}, index = taxa_data.columns)
features_scores_tree_based.sort_values("XGboost_average_feature_importance", ascending=False, inplace=True)

#Create a plot with the desired number of highest scoring features (e.g. 30):
plot_feature_importance_scores(features_scores_tree_based, 40)

#Create a new table with only a subset of the original features, based on empirical number of highest scoring features (user selected) 
taxa_data_sub = subset_features_in_df(features_scores_tree_based.index[0:60], taxa_data)
#The highest scoring 40 features (species) obtained in the original manuscript:
best_features_species_40 = pd.read_csv("/home/dnx/Downloads/ABX_gut_metagenome/Pouchitis_classifiers/Species_average_features_importance_XGBoost_5CV-50rep.txt", index_col=0, sep="\t")
taxa_data_sub = subset_features_in_df(best_features_species_40.index, taxa_data)

###------------------------------------------------------------------------------------###

###----Grid Search Hyperparameters Tuning for XGboost model----###
#The most commonly tunable parameters for XGBoost:
reg_alpha = np.array([1,0.1,0.5,0.01,0.001,0.0001,0])
reg_lambda = np.array([1,0.1,0.5,0.01,0.001,0.0001,0])
gamma = np.arange(0,1.1,0.1)
max_depth = np.arange(1,11,1)
n_estimators = np.arange(50,550,50)
colsample_bytree = np.arange(0.1,1.1,0.1)
subsample = np.arange(0.1,1.1,0.1)
min_child_weight = np.array([1,2,3,5,6,7])
#learning rate = [2-10]/# trees:
learning_rate = 5/np.arange(50,550,50)
#choose the desired combination of parameters to run grid search on. Beware that running time increases significantly as more complex combinations are tried.
model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree=0.5, subsample=0.5, random_state=10)
grid = GridSearchCV(estimator=model, param_grid=dict(n_estimators=n_estimators, learning_rate=learning_rate),
                    cv=5, scoring='roc_auc', n_jobs=2) #dict(max_depth=max_depth, n_estimators=n_estimators, colsample_bytree=colsample_bytree, subsample=subsample)
grid.fit(taxa_data_po, target_labels["labels_binary"])
#summarize the results of the grid search
print("Best score: %0.3f using parameter %s" % (grid.best_score_, grid.best_params_))  
###------------------------------------------------------------###


#----------------------Utility functions:-----------------------------------###

def subset_features_in_df (features_list, df):
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

def plot_feature_importance_scores (df, number_features):
    '''Provide a table with feature names and importance scores (df) and the number of features you wish to plot (number_features)'''
    features_plot = features_scores_tree_based.iloc[0:number_features,:]
    colors_map = {"commensal - unknown" : "darkblue","potential pathobiont / oral" : "darkred", "beneficial" : "forestgreen"}
    plt.figure(figsize=(number_features*0.3, 5))
    ax = features_plot["XGboost_average_feature_importance"].plot.bar(color="darkred",linewidth=0.1, width=0.5, legend=False, edgecolor = "darkblue") 
    plt.xlabel("Features used for prediction", fontsize=18, fontweight='bold')
    plt.ylabel("Average feature importance scores", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, rotation=90)

###-------------------------------------------------------------------------###
