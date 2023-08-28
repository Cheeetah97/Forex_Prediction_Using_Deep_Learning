import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
tf.keras.utils.set_random_seed(0)

#%%
# TimeSeries Scaler Class
class TSScaler(BaseEstimator,TransformerMixin):
    def __init__(self,range=(0,1)):
        self.range=range
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=self.range)
    
    def fit(self,X,y=None):
        self.standard_scaler.fit(X)
        standardized = self.standard_scaler.transform(X)
        self.minmax_scaler.fit(standardized)
        return self
    
    def transform(self,X):
        standardized = self.standard_scaler.transform(X)
        scaled = self.minmax_scaler.transform(standardized)
        return scaled
    
    def inverse_transform(self,X):
        standardized = self.minmax_scaler.inverse_transform(X)
        unscaled = self.standard_scaler.inverse_transform(standardized)
        return unscaled

#%%
# Features
def difference_features(df,Target):
    feats = []

    # Price Differences
    for dif in range(1,4):
        df[f'{Target}_diff_{dif}'] = df[Target].diff(dif)
        feats.append(f'{Target}_diff_{dif}')
    
    # First Difference Rolling Sum and Std
    for window in [2,3,4]:
        #df[f'{Target}_rollmea{window}_1'] = df[Target].diff(1).rolling(window,min_periods=1).sum()
        df[f'{Target}_rollstd{window}_1'] = df[Target].diff(1).rolling(window,min_periods=1).std() 
        #feats.append(f'{Target}_rollmea{window}_1')
        feats.append(f'{Target}_rollstd{window}_1')

    return df

def return_features(df,Target):
    feats = []

    # Price Return Lags
    for lag in range(2,4):
        df[f'{Target}_lag_{lag}'] = df[Target].shift(lag)
        feats.append(f'{Target}_lag_{lag}')
    
    # First Lag Rolling Sum and Std
    for window in [2,3,4]:
        df[f'{Target}_rollmea{window}_1'] = df[Target].rolling(window,min_periods=1).sum()
        df[f'{Target}_rollstd{window}_1'] = df[Target].rolling(window,min_periods=1).std()
        feats.append(f'{Target}_rollmea{window}_1')
        feats.append(f'{Target}_rollstd{window}_1')

    return df

def relative_strength_index(df,Target):
    delta = df[Target]
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(14).mean()
    avg_loss = abs(loss.rolling(14).mean())
    rs = avg_gain / avg_loss
    df[Target+'_rsi'] = 100 - (100 / (1 + rs))
    return df

#%%
# Test Set Generation
def generate_test_portions(df,portion_length,num_test_portions):
    np.random.seed(0)
    length = len(df)
    test_portions = []
    for _ in range(num_test_portions):
        start = np.random.randint(0, length - portion_length)
        end = start + portion_length
        test_portion = df.loc[start:end,:].index
        test_portions.append(test_portion)
    return test_portions

#%%
# Features and Targets Scaling
def x_scaler(x_train,x_valid,x_test,scaler):

    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    x_test_scaled = []
    for i in range(len(x_test)):
        x_test_s = scaler.transform(x_test[i])
        x_test_scaled.append(x_test_s)
    
    return x_train_scaled,x_valid_scaled,x_test_scaled

def y_scaler(y_train,y_valid,y_test,scaler):

    scaler.fit(y_train)

    y_train_scaled = scaler.transform(y_train)
    y_valid_scaled = scaler.transform(y_valid)

    y_test_scaled = []
    for i in range(len(y_test)):
        y_test_s = scaler.transform(y_test[i])
        y_test_scaled.append(y_test_s)
    
    with open('scaler_y'+'.pkl','wb') as file:
            pickle.dump(scaler,file)
    
    return y_train_scaled,y_valid_scaled,y_test_scaled

#%%
# Get K-Folds for Cross-Validation
def get_folds(x,y,df,folds,selected_folds):
    kf = KFold(n_splits=folds,shuffle=True)
    kf = kf.split(df)

    fold_count = 0
    train_portions_x = []
    train_portions_y = []
    valid_portions_x = []
    valid_portions_y = []
    
    for fold,(train_idx,valid_idx) in enumerate(kf):

        if fold_count >= selected_folds:
            break

        X_train = x[train_idx,:]
        Y_train = y[train_idx,:]

        X_valid = x[valid_idx,:]
        Y_valid = y[valid_idx,:]

        train_portions_x.append(X_train)
        train_portions_y.append(Y_train)

        valid_portions_x.append(X_valid)
        valid_portions_y.append(Y_valid)

        fold_count += 1
    
    return train_portions_x,train_portions_y,valid_portions_x,valid_portions_y

#%%
# Clipping Target Values
def target_clip(arr,upper,lower):
    for i in range(arr.shape[1]):
        column = arr[:, i]
    
        values_greater_than_0 = column[column > 0]
        values_less_than_0 = column[column < 0]
    
        p_up = np.percentile(values_greater_than_0,upper)
        p_low = np.percentile(values_less_than_0,lower)
    
        column[column > 0] = np.clip(column[column > 0],None,p_up)
        column[column < 0] = np.clip(column[column < 0],None,p_low)
        arr[:, i] = column
    return arr

#%%
# Compiling the Predictions
def compile_predictions(preds,n_vars,scaler_objs,col_names):
    
    for i in range(n_vars):
        if i == 0:
            pred_df = pd.DataFrame(data=preds[i],columns=[col_names[i]])
        else:
            pred_df = pd.concat([pred_df,pd.DataFrame(data=preds[i],columns=[col_names[i]])],axis=1)
            
    pred_df = pd.DataFrame(scaler_objs.inverse_transform(pred_df),columns=list(pred_df.columns))
    return pred_df

#%%
# Feature Selection Using Recursive Feature Elimination

def select_features(x,y,estimator):
    selected_features = []
    for t in range(y.shape[1]):
        rfecv = RFECV(estimator=estimator,step=5,cv=KFold(n_splits=10),scoring='neg_mean_squared_error',n_jobs=-1)
        rfecv.fit(x,y[:,t])
        selected_features.append(rfecv.support_)
    return selected_features