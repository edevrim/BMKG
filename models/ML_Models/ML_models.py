#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 01:21:36 2019

@author: salihemredevrim
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import math 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
#%% embeddings are added as variables to user-book pairs in training and test set
#In user_embeddings only training info used which is like a user liked a book if s/he gave it more than 3
#In book_embeddings only general tags for books are used 
# Run once 

X_train = pd.read_excel('X_train.xlsx').reset_index(drop=False)
X_test = pd.read_excel('X_test.xlsx').reset_index(drop=False)
y_train = pd.read_excel('y_train.xlsx').reset_index(drop=False)
y_test = pd.read_excel('y_test.xlsx').reset_index(drop=False)
user_embeddings_sg = pd.read_excel('user_embeddings_sg.xlsx').reset_index(drop=True)
book_embeddings_sg = pd.read_excel('book_embeddings_sg.xlsx').reset_index(drop=True)

user_embeddings_cbow = pd.read_excel('user_embeddings_cbow.xlsx').reset_index(drop=True)
book_embeddings_cbow = pd.read_excel('book_embeddings_cbow.xlsx').reset_index(drop=True)

#skip-gram embeddings 
#If a user or book has never been in training then she or he or it doesn't have embeddings, so ignoring them by inner join
X_train_sg = pd.merge(X_train, book_embeddings_sg, how='inner', on='goodreads_book_id')
X_train_sg = pd.merge(X_train_sg, user_embeddings_sg, how='inner', on='user_id')
#
X_test_sg = pd.merge(X_test, book_embeddings_sg, how='inner', on='goodreads_book_id')
X_test_sg = pd.merge(X_test_sg, user_embeddings_sg, how='inner', on='user_id')

#cbow embeddings
X_train_cbow = pd.merge(X_train, book_embeddings_cbow, how='inner', on='goodreads_book_id')
X_train_cbow = pd.merge(X_train_cbow, user_embeddings_cbow, how='inner', on='user_id')
#
X_test_cbow = pd.merge(X_test, book_embeddings_cbow, how='inner', on='goodreads_book_id')
X_test_cbow = pd.merge(X_test_cbow, user_embeddings_cbow, how='inner', on='user_id')

y_train = y_train[y_train['index'].isin(X_train_sg['index'])].reset_index(drop=True)
y_test = y_test[y_test['index'].isin(X_test_sg['index'])].reset_index(drop=True)

X_train_cbow = X_train_cbow.reset_index(drop=True)
X_train_sg = X_train_sg.reset_index(drop=True)
X_test_cbow = X_test_cbow.reset_index(drop=True)
X_test_sg = X_test_sg.reset_index(drop=True)

#%%
del user_embeddings_sg, book_embeddings_sg, user_embeddings_cbow, book_embeddings_cbow 

#%%dot product of user_embeddings and book_embeddings 

def dotprod(data1, data_for_missing, word2vecsize, type1):
    len1 = len(data1)
    len1 = math.ceil(len1/10000) 
    #filling missing values with mean of the train set
    data1 = data1.fillna(data_for_missing.mean());
    data1[type1] = 0; 
    for k in range(len1):
        print(k)
        xd1 = data1.iloc[k*10000:(k+1)*10000, 3:word2vecsize+3]
        xd2 = data1.iloc[k*10000:(k+1)*10000, word2vecsize+3:-1]
        data1[type1].iloc[k*10000:(k+1)*10000] = np.dot(xd1, xd2.T).diagonal()
    return data1 
#        
#%% Run once
    
word2vecsize = 50; 
 
X_test_sg = dotprod(X_test_sg, X_train_sg, word2vecsize, 'sg')    
X_train_sg = dotprod(X_train_sg, X_train_sg, word2vecsize, 'sg')   

X_test_cbow = dotprod(X_test_cbow, X_train_cbow, word2vecsize, 'cbow')    
X_train_cbow = dotprod(X_train_cbow, X_train_cbow, word2vecsize, 'cbow')      

#%%last join and sort before we go
X_train = pd.merge(X_train_sg, X_train_cbow, how='left', on=['user_id', 'goodreads_book_id', 'index']).sort_values('index').reset_index(drop=True)
X_test = pd.merge(X_test_sg, X_test_cbow, how='left', on=['user_id', 'goodreads_book_id', 'index']).sort_values('index').reset_index(drop=True)

y_train = y_train.sort_values('index').reset_index(drop=True)
y_test = y_test.sort_values('index').reset_index(drop=True)
#%%

del X_test_cbow, X_train_cbow, X_test_sg, X_train_sg, word2vecsize

#%%
#Variable selection 

def var_select(train_X, train_y, target1, corr_cut1):
  
    train_X = train_X.drop(['goodreads_book_id', 'user_id', 'index'], axis=1);
    train_y = pd.DataFrame(train_y[target1]);
    
    #filling missing values with mean of the train set
    #train_X = train_X.fillna(train_X.mean());
    
    #correlation check 
    corr_matrix = train_X.corr().abs();
    
    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool));
    
    # Find index of feature columns with correlation greater than corr_cut1
    to_drop = [column for column in upper.columns if any(upper[column] > corr_cut1)];
    
    train_X = train_X.drop(to_drop, axis=1); 
    
    #Correlation between variables and target 
    features = train_X.columns.tolist();
    correlations = {};
    for f in features:
        data_temp = pd.concat([train_X[f], train_y[target1]], axis=1);
        x1 = data_temp[f].values;
        x2 = data_temp[target1].values;
        key = f + ' vs ' + 'target';
        correlations[key] = pearsonr(x1,x2)[0];
    
    data_correlations = pd.DataFrame(correlations, index=['Value']).T;
    
    #Based on poor target explanation
    to_drop2 = data_correlations[data_correlations['Value'].abs() < 0.025].reset_index()
    to_drop2 = to_drop2['index'].str.split(' ', 1).str[0]
    to_drop2 = to_drop2.tolist()
    
    return to_drop, to_drop2, data_correlations; 

#%%
to_drop5, to_drop2_5, data_correlations5 = var_select(X_train, y_train, 'rating_binary5', 0.9)
to_drop45, to_drop2_45, data_correlations45 = var_select(X_train, y_train, 'rating_binary45', 0.9)

##test check
#dot products don't work well on test set
#to_drop51, to_drop2_51, data_correlations51 = var_select(X_test, y_test, 'rating_binary5', 0.9)
#to_drop451, to_drop2_451, data_correlations451 = var_select(X_test, y_test, 'rating_binary45', 0.9)

X_train_5 = X_train.drop(to_drop2_5, axis=1)
X_test_5 = X_test.drop(to_drop2_5, axis=1)

X_train_45 = X_train.drop(to_drop2_45, axis=1)
X_test_45 = X_test.drop(to_drop2_45, axis=1)

#also dot products are dropped 
X_train_5 = X_train_5.drop(['cbow', 'sg'], axis=1)
X_test_5 = X_test_5.drop(['cbow', 'sg'], axis=1)

X_train_45 = X_train_45.drop(['cbow', 'sg'], axis=1)
X_test_45 = X_test_45.drop(['cbow', 'sg'], axis=1)

#%%
del X_train, X_test

#%%

#classification models
# 1: Random Forest with random search with kfold 
def rf_go(X_train, y_train, X_test, y_test, target1, iter1, k): 
   
   #no need variable selection but added to make it faster 
#   to_drop, data_correlations = var_select(X_train, y_train, target1, corr_cut1)
#   print('variable selection is done!')
#   
   X_train = X_train.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_train = X_train.drop((to_drop), axis=1)

   X_test = X_test.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_test = X_test.drop((to_drop), axis=1)
  
   #filling missing values of both with mean of the train set
   X_train = X_train.fillna(X_train.mean()); 
   X_test = X_test.fillna(X_train.mean()); 
   
   y_train = pd.DataFrame(y_train[target1])
   y_test = pd.DataFrame(y_test[target1])
   print('data prep is done!')  
   
   model = RandomForestClassifier()

#n_estimators = number of trees in the foreset
#max_features = max number of features considered for splitting a node
#max_depth = max number of levels in each decision tree
#min_samples_split = The minimum number of samples required to split an internal node
   
   row1, col1 = X_train.shape
   
   cv1= KFold(n_splits=k, shuffle=True, random_state=1)
    
   param_grid = {'n_estimators': [50, 75, 100],
                  'max_features': [min(15, col1), min(25, col1), min(50, col1), min(70, col1)], 
                  'max_depth': [4, 5, 6], 
                  'min_samples_split': [15, 30, 50, 70]
                  };
    
   grid = RandomizedSearchCV(model, param_grid, random_state=1905, n_iter=iter1, cv=cv1, verbose=1, scoring='roc_auc', n_jobs=-1)
     
   #fit grid search
   model1 = grid.fit(X_train, y_train.values.ravel())
   print('grid search is done!')  
   grid_best_score = model1.best_score_ 
   print(model1.best_score_) 
    
   ##take best estimator then fit on train then test on test set
   model2 = model1.best_estimator_.fit(X_train, y_train.values.ravel())
   pred_test = pd.DataFrame(model2.predict(X_test)).set_index(X_test.index)
   pred_test_proba = pd.DataFrame(model2.predict_proba(X_test)).set_index(X_test.index) 
   pred_test_proba = pd.DataFrame(pred_test_proba[1])
   pred_test.columns = [target1]
   pred_test_proba.columns = [target1] 

   #performance
   roc_test = roc_auc_score(y_test, pred_test_proba)
   accuracy_test = accuracy_score(y_test, pred_test)
   confusion = confusion_matrix(y_test, pred_test)
   
   print('scoring is done!')  
   
   #save
   writer = pd.ExcelWriter(target1+'_pred_test_RF.xlsx', engine='xlsxwriter');
   pred_test.to_excel(writer, sheet_name= 'pred_test');
   writer.save();
   
   writer = pd.ExcelWriter(target1+'_pred_test_proba_RF.xlsx', engine='xlsxwriter');
   pred_test_proba.to_excel(writer, sheet_name= 'pred_test_proba');
   writer.save();
   
  #filename = 'rf_model.sav'
  #pickle.dump(model2, open(filename, 'wb'))
   
   TP = confusion[0][0]; FN = confusion[0][1]; FP=confusion[1][0]; TN=confusion[1][1];
   
   output = {
         "grid_best_score": grid_best_score,    
		"roc_test" : roc_test,
         "accuracy_test" : accuracy_test, 
         "TP" : TP, 
         "FN" : FN, 
         "FP" : FP, 
         "TN" : TN
	}
   
   output1 = pd.DataFrame(output, index = [0])
   
   #save
   writer = pd.ExcelWriter(target1+'_rf_output.xlsx', engine='xlsxwriter');
   output1.to_excel(writer, sheet_name= 'output');
   writer.save();

   return output; 

#%%    

# 2: XGBoost with random search with kfold 

#https://xgboost.readthedocs.io/en/latest/parameter.html
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

def xgb_go(X_train, y_train, X_test, y_test, target1, iter1, k): 
    
#   #no need variable selection but added to make it faster 
#   to_drop, data_correlations = var_select(X_train, y_train, target1, corr_cut1)
#   print('variable selection is done!')
    
   X_train = X_train.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_train = X_train.drop((to_drop), axis=1)

   X_test = X_test.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_test = X_test.drop((to_drop), axis=1)
  
   #filling missing values of both with mean of the train set
   X_train = X_train.fillna(X_train.mean()); 
   X_test = X_test.fillna(X_train.mean()); 
   
   y_train = pd.DataFrame(y_train[target1])
   y_test = pd.DataFrame(y_test[target1])
   print('data prep is done!')  

   model = XGBClassifier();
   
   cv1= KFold(n_splits=k, shuffle=True, random_state=1)
    
   param_grid = {
        'silent': [False],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.1, 0.5],
        'subsample': [0.25, 0.5, 0.75],
        'min_child_weight': [4.0, 5.0, 6.0, 7.0],
        'gamma': [1.0, 2.0, 3.0],
        'n_estimators': [50, 75, 100]}
    
   grid = RandomizedSearchCV(model, param_grid, random_state=1905, n_iter=iter1, cv=cv1, verbose=1, scoring='roc_auc', n_jobs=-1)
   
   #fit grid search
   model1 = grid.fit(X_train, y_train.values.ravel())
   print('grid search is done!')   
   grid_best_score = model1.best_score_ 
   print(model1.best_score_)
    
   ##take best estimator then fit on train then test on test set
   model2 = model1.best_estimator_.fit(X_train, y_train.values.ravel())
   pred_test = pd.DataFrame(model2.predict(X_test)).set_index(X_test.index)
   pred_test_proba = pd.DataFrame(model2.predict_proba(X_test)).set_index(X_test.index) 
   pred_test_proba = pd.DataFrame(pred_test_proba[1])
   pred_test.columns = [target1]
   pred_test_proba.columns = [target1] 

   #performance
   roc_test = roc_auc_score(y_test, pred_test_proba)
   accuracy_test = accuracy_score(y_test, pred_test)
   confusion = confusion_matrix(y_test, pred_test)
   
   print('scoring is done!')  
   #save
   writer = pd.ExcelWriter(target1+'_pred_test_XGB.xlsx', engine='xlsxwriter');
   pred_test.to_excel(writer, sheet_name= 'pred_test');
   writer.save();
   
   writer = pd.ExcelWriter(target1+'_pred_test_proba_XGB.xlsx', engine='xlsxwriter');
   pred_test_proba.to_excel(writer, sheet_name= 'pred_test_proba');
   writer.save();
   
  # filename = 'xgb_model.sav'
  # pickle.dump(model2, open(filename, 'wb'))
   
   TP = confusion[0][0]; FN = confusion[0][1]; FP=confusion[1][0]; TN=confusion[1][1];
   
   output = {
         "grid_best_score": grid_best_score,    
		"roc_test" : roc_test,
         "accuracy_test" : accuracy_test, 
         "TP" : TP, 
         "FN" : FN, 
         "FP" : FP, 
         "TN" : TN
	}
   
   output1 = pd.DataFrame(output, index = [0])
   
   #save
   writer = pd.ExcelWriter(target1+'_xgb_output.xlsx', engine='xlsxwriter');
   output1.to_excel(writer, sheet_name= 'output');
   writer.save();
   
   return output; 

#%%
   
# 3: Logistic Regression with random search with kfold 

def logistic_go(X_train, y_train, X_test, y_test, target1, iter1, k): 
    
#   #no need variable selection but added to make it faster 
#   to_drop, data_correlations = var_select(X_train, y_train, target1, corr_cut1)
#   print('variable selection is done!')
   
   X_train = X_train.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_train = X_train.drop((to_drop), axis=1)
#
   X_test = X_test.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_test = X_test.drop((to_drop), axis=1)
  
   #filling missing values of both with mean of the train set
   X_train = X_train.fillna(X_train.mean()); 
   X_test = X_test.fillna(X_train.mean()); 
   
   y_train = pd.DataFrame(y_train[target1])
   y_test = pd.DataFrame(y_test[target1])
   print('data prep is done!')  
   
   model = LogisticRegression();
   
   cv1= KFold(n_splits=k, shuffle=True, random_state=1)
    
   #Create regularization penalty space
   penalty = ['l1', 'l2']

   # Create regularization hyperparameter distribution using uniform distribution
   C = [0.001, 0.025, 0.01, 0.1, 1, 10, 100]

   # Create hyperparameter options
   hyperparameters = dict(C=C, penalty=penalty)
    
   grid = RandomizedSearchCV(model, hyperparameters, random_state=1905, n_iter=iter1, cv=cv1, verbose=1, scoring = 'roc_auc', n_jobs=-1)

   #fit grid search
   model1 = grid.fit(X_train, y_train.values.ravel())
   print('grid search is done!')  
   grid_best_score = model1.best_score_ 
   print(model1.best_score_) 
   print(model1.best_params_) 
    
   ##take best estimator then fit on train then test on test set
   model2 = model1.best_estimator_.fit(X_train, y_train.values.ravel())
   pred_test = pd.DataFrame(model2.predict(X_test)).set_index(X_test.index)
   pred_test_proba = pd.DataFrame(model2.predict_proba(X_test)).set_index(X_test.index) 
   pred_test_proba = pd.DataFrame(pred_test_proba[1])
   pred_test.columns = [target1]
   pred_test_proba.columns = [target1] 

   #performance
   roc_test = roc_auc_score(y_test, pred_test_proba)
   accuracy_test = accuracy_score(y_test, pred_test)
   confusion = confusion_matrix(y_test, pred_test)
   
   print('scoring is done!')  
   #save
   writer = pd.ExcelWriter(target1+'_pred_test_LOG.xlsx', engine='xlsxwriter');
   pred_test.to_excel(writer, sheet_name= 'pred_test');
   writer.save();
   
   writer = pd.ExcelWriter(target1+'_pred_test_proba_LOG.xlsx', engine='xlsxwriter');
   pred_test_proba.to_excel(writer, sheet_name= 'pred_test_proba');
   writer.save();
   
 #  filename = 'logr_model.sav'
 #  pickle.dump(model2, open(filename, 'wb'))
   
   TP = confusion[0][0]; FN = confusion[0][1]; FP=confusion[1][0]; TN=confusion[1][1];
   
   output = {
         "grid_best_score": grid_best_score,    
		"roc_test" : roc_test,
         "accuracy_test" : accuracy_test, 
         "TP" : TP, 
         "FN" : FN, 
         "FP" : FP, 
         "TN" : TN
	}
   
   output1 = pd.DataFrame(output, index = [0])
   
   #save
   writer = pd.ExcelWriter(target1+'_logr_output.xlsx', engine='xlsxwriter');
   output1.to_excel(writer, sheet_name= 'output');
   writer.save();
   
   return output; 
   
#%% 
# 4: SVM with random search with kfold 
   
def svm_go(X_train, y_train, X_test, y_test, target1, iter1, k): 
    
#   #no need variable selection but added to make it faster 
#   to_drop, data_correlations = var_select(X_train, y_train, target1, corr_cut1)
#   print('variable selection is done!')
   
   X_train = X_train.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_train = X_train.drop((to_drop), axis=1)
#
   X_test = X_test.drop(['goodreads_book_id', 'user_id', 'index'], axis=1)
#   X_test = X_test.drop((to_drop), axis=1)
  
   #filling missing values of both with mean of the train set
   X_train = X_train.fillna(X_train.mean()); 
   X_test = X_test.fillna(X_train.mean()); 
   
   y_train = pd.DataFrame(y_train[target1])
   y_test = pd.DataFrame(y_test[target1])
   print('data prep is done!')  
   
   model = SVC();
   
   cv1= KFold(n_splits=k, shuffle=True, random_state=1)
    
   param_grid = {'kernel':['linear'], 'C':[ 1,0.5,0.75], 'gamma': [1,2], 'probability': [True]}
    
   grid = RandomizedSearchCV(model, param_grid, random_state=1905, n_iter=iter1, cv=cv1, verbose=1, scoring='roc_auc', n_jobs=-1)
    
   #fit grid search
   model1 = grid.fit(X_train, y_train.values.ravel())
   print('grid search is done!')  
   grid_best_score = model1.best_score_ 
   print(model1.best_score_) 
    
   ##take best estimator then fit on train then test on test set
   model2 = model1.best_estimator_.fit(X_train, y_train.values.ravel())
   pred_test = pd.DataFrame(model2.predict(X_test)).set_index(X_test.index)
   pred_test_proba = pd.DataFrame(model2.predict_proba(X_test)).set_index(X_test.index) 
   pred_test_proba = pd.DataFrame(pred_test_proba[1])
   pred_test.columns = [target1]
   pred_test_proba.columns = [target1] 

   #performance
   roc_test = roc_auc_score(y_test, pred_test_proba)
   accuracy_test = accuracy_score(y_test, pred_test)
   confusion = confusion_matrix(y_test, pred_test)
   
   print('scoring is done!')  
   #save
   writer = pd.ExcelWriter(target1+'_pred_test_SVM.xlsx', engine='xlsxwriter');
   pred_test.to_excel(writer, sheet_name= 'pred_test');
   writer.save();
   
   writer = pd.ExcelWriter(target1+'_pred_test_proba_SVM.xlsx', engine='xlsxwriter');
   pred_test_proba.to_excel(writer, sheet_name= 'pred_test_proba');
   writer.save();
   
  # filename = 'svm_model.sav'
  # pickle.dump(model2, open(filename, 'wb'))
   
   TP = confusion[0][0]; FN = confusion[0][1]; FP=confusion[1][0]; TN=confusion[1][1];
   
   output = {
         "grid_best_score": grid_best_score,    
		"roc_test" : roc_test,
         "accuracy_test" : accuracy_test, 
         "TP" : TP, 
         "FN" : FN, 
         "FP" : FP, 
         "TN" : TN
	}
   
   output1 = pd.DataFrame(output, index = [0])
   
   #save
   writer = pd.ExcelWriter(target1+'_svm_output.xlsx', engine='xlsxwriter');
   output1.to_excel(writer, sheet_name= 'output');
   writer.save();
   
   return output; 

#%%

logr_output5 = logistic_go(X_train_5, y_train, X_test_5, y_test, 'rating_binary5', 14, 3);
logr_output45 = logistic_go(X_train_45, y_train, X_test_45, y_test, 'rating_binary45', 14, 3);

rf_output5 = rf_go(X_train_5, y_train, X_test_5, y_test, 'rating_binary5', 50, 3);
rf_output45 = rf_go(X_train_45, y_train, X_test_45, y_test, 'rating_binary45', 50, 3);

xgb_output5 = xgb_go(X_train_5, y_train, X_test_5, y_test, 'rating_binary5', 20, 3);
xgb_output45 = xgb_go(X_train_45, y_train, X_test_45, y_test, 'rating_binary45', 20, 3);

svm_output5 = svm_go(X_train_5, y_train, X_test_5, y_test, 'rating_binary5', 3, 3);
svm_output45 = svm_go(X_train_45, y_train, X_test_45, y_test, 'rating_binary45', 3, 3);

#%%
#Dot product analysis (ignored)
#
##init
#X_Train_1 = X_train[['index', 'user_id', 'sg', 'cbow']]
#X_Test_1 = X_test[['index', 'user_id', 'sg', 'cbow']]
#
#Collect_all = pd.DataFrame(X_Train_1['user_id'].unique())
#Collect_all.columns = ['user_id']
#Collect_all['train_size'] = 0
#Collect_all['test_size'] = 0
#Collect_all['roc_test'] = 0
#Collect_all['accuracy_test'] = 0
#
#Collect_all = Collect_all.set_index('user_id')
#
#def dot_analysis(target1): 
#    counter = 0;
#    model = DecisionTreeClassifier(max_depth=1)
#    s = Collect_all.reset_index()
#    s = s.user_id
#    
#    for k in s.tolist():
#        
#        counter = counter + 1;
#        print(k)
#        print(counter)
#        
#        X_Train_2 = X_Train_1[X_Train_1['user_id'] == np.int64(k)].reset_index(drop=True)
#        X_Test_2 = X_Test_1[X_Test_1['user_id'] == np.int64(k)].reset_index(drop=True)
#
#        len_train = len(X_Train_2)
#        len_test = len(X_Test_2)
#
#        y_Train_2 = y_train[y_train['index'].isin(X_Train_2['index'])].reset_index(drop=True)
#        y_Test_2 = y_test[y_test['index'].isin(X_Test_2['index'])].reset_index(drop=True)
#        
#        y_Train_2 = pd.DataFrame(y_Train_2[target1])
#        y_Test_2 = pd.DataFrame(y_Test_2[target1])
#        
#        #for check
#        avg1 = y_Train_2.mean().get_value(0)
#        avg2 = y_Test_2.mean().get_value(0)
#
#        X_Train_2 = X_Train_2[['sg', 'cbow']]
#        X_Test_2 = X_Test_2[['sg', 'cbow']]
#
#        if avg1 > 0.0 and avg1 < 1.0 and avg2 > 0.0 and avg2 < 1.0:
#            
#            model1 = model.fit(X_Train_2, y_Train_2.values.ravel())
#    
#            pred_test = pd.DataFrame(model1.predict(X_Test_2)).set_index(X_Test_2.index)
#            pred_test_proba = pd.DataFrame(model1.predict_proba(X_Test_2)).set_index(X_Test_2.index) 
#            pred_test_proba = pd.DataFrame(pred_test_proba[1])
#            pred_test.columns = [target1]
#            pred_test_proba.columns = [target1] 
#    
#            #performance
#            roc_test = roc_auc_score(y_Test_2, pred_test_proba)
#            accuracy_test = accuracy_score(y_Test_2, pred_test)
#
#            Collect_all['train_size'].loc[k] = len_train
#            Collect_all['test_size'].loc[k]  = len_test 
#            Collect_all['roc_test'].loc[k] = roc_test
#            Collect_all['accuracy_test'].loc[k] = accuracy_test
#
#    return Collect_all;
#
##%%   
#Coll_5 = dot_analysis('rating_binary5');
#Coll_45 = dot_analysis('rating_binary45');
#
##save
#writer = pd.ExcelWriter('Coll_5.xlsx', engine='xlsxwriter');
#Coll_5.to_excel(writer, sheet_name= 'xd');
#writer.save();
#   
#writer = pd.ExcelWriter('Coll_45.xlsx', engine='xlsxwriter');
#Coll_45.to_excel(writer, sheet_name= 'xd');
#writer.save();
#
#Coll_45_considered = Coll_45[Coll_45['train_size'] >= 1]
#plt.hist(x=Coll_45_considered['roc_test'])
#plt.show()
#
#Coll_45_considered = Coll_45[Coll_45['train_size'] >= 1]
#plt.hist(x=Coll_45_considered['accuracy_test'])
#plt.show()
#
#Coll_5_considered = Coll_5[Coll_5['train_size'] >= 1]
#plt.hist(x=Coll_5_considered['roc_test'])
#plt.show()
#
#
#Coll_5_considered = Coll_5[Coll_5['train_size'] >= 1]
#plt.hist(x=Coll_5_considered['accuracy_test'])
#plt.show()