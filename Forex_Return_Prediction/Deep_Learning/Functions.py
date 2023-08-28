import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import tensorflow_probability as tfp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
# Additive Attention Layer
class attention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x,self.W)+self.b)
        e = tf.keras.backend.squeeze(e,axis=-1)   
        alpha = tf.keras.backend.softmax(e,axis=-1)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = x * alpha
        return context

#%%
# Defining RNN
def RNN(window_size,n_cols,n_neurons,n_layers,activation,loss,optimizer):

    if n_layers == 1:
        # Inputs
        inputs= tf.keras.Input(shape=(window_size,n_cols))

        # GRU Layers
        l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation)
        l_1_outputs = l_1(inputs)
        l_1_outputs = tf.keras.layers.Dropout(0.5)(l_1_outputs)

        # Outputs
        dense = tf.keras.layers.Dense(4,activation=activation)
        dense_outputs = dense(l_1_outputs)

        model = tf.keras.models.Model([inputs],[dense_outputs])

        # Compiling the RNN
        model.compile(optimizer=optimizer,loss=loss)
        return model
    
    elif n_layers == 2:
        # Inputs
        inputs= tf.keras.Input(shape=(window_size,n_cols))

        # GRU Layers
        l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_1_outputs = l_1(inputs)
        l_1_outputs = tf.keras.layers.Dropout(0.5)(l_1_outputs)

        l_2 = tf.keras.layers.LSTM(int(n_neurons),activation=activation)
        l_2_outputs = l_2(l_1_outputs)
        l_2_outputs = tf.keras.layers.Dropout(0.5)(l_2_outputs)

        # Outputs
        dense = tf.keras.layers.Dense(4,activation=activation)
        dense_outputs = dense(l_2_outputs)

        model = tf.keras.models.Model([inputs],[dense_outputs])

        # Compiling the RNN
        model.compile(optimizer=optimizer,loss=loss)
        return model


    elif n_layers == 3:
        # Inputs
        inputs= tf.keras.Input(shape=(window_size,n_cols))

        # GRU Layers
        l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_1_outputs = l_1(inputs)
        l_1_outputs = tf.keras.layers.Dropout(0.5)(l_1_outputs)

        l_2 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_2_outputs = l_2(l_1_outputs)
        l_2_outputs = tf.keras.layers.Dropout(0.5)(l_2_outputs)

        l_3 = tf.keras.layers.LSTM(int(n_neurons),activation=activation)
        l_3_outputs = l_3(l_2_outputs)
        l_3_outputs = tf.keras.layers.Dropout(0.5)(l_3_outputs)

        # Outputs
        dense = tf.keras.layers.Dense(4,activation=activation)
        dense_outputs = dense(l_3_outputs)

        model = tf.keras.models.Model([inputs],[dense_outputs])

        # Compiling the RNN
        model.compile(optimizer=optimizer,loss=loss)
        return model

    elif n_layers == 4:
        # Inputs
        inputs= tf.keras.Input(shape=(window_size,n_cols))

        # GRU Layers
        l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_1_outputs = l_1(inputs)
        l_1_outputs = tf.keras.layers.Dropout(0.5)(l_1_outputs)

        l_2 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_2_outputs = l_2(l_1_outputs)
        l_2_outputs = tf.keras.layers.Dropout(0.5)(l_2_outputs)

        l_3 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_3_outputs = l_3(l_2_outputs)
        l_3_outputs = tf.keras.layers.Dropout(0.5)(l_3_outputs)

        l_4 = tf.keras.layers.LSTM(int(n_neurons),activation=activation)
        l_4_outputs = l_4(l_3_outputs)
        l_4_outputs = tf.keras.layers.Dropout(0.5)(l_4_outputs)

        # Outputs
        dense = tf.keras.layers.Dense(4,activation=activation)
        dense_outputs = dense(l_4_outputs)

        model = tf.keras.models.Model([inputs],[dense_outputs])

        # Compiling the RNN
        model.compile(optimizer=optimizer,loss=loss)
        return model
    
    elif n_layers == 5:
        # Inputs
        inputs= tf.keras.Input(shape=(window_size,n_cols))

        # GRU Layers
        l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_1_outputs = l_1(inputs)
        l_1_outputs = tf.keras.layers.Dropout(0.5)(l_1_outputs)

        l_2 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_2_outputs = l_2(l_1_outputs)
        l_2_outputs = tf.keras.layers.Dropout(0.5)(l_2_outputs)

        l_3 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_3_outputs = l_3(l_2_outputs)
        l_3_outputs = tf.keras.layers.Dropout(0.5)(l_3_outputs)

        l_4 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_4_outputs = l_4(l_3_outputs)
        l_4_outputs = tf.keras.layers.Dropout(0.5)(l_4_outputs)

        l_5 = tf.keras.layers.LSTM(int(n_neurons),activation=activation)
        l_5_outputs = l_5(l_4_outputs)
        l_5_outputs = tf.keras.layers.Dropout(0.5)(l_5_outputs)

        # Outputs
        dense = tf.keras.layers.Dense(4,activation=activation)
        dense_outputs = dense(l_5_outputs)

        model = tf.keras.models.Model([inputs],[dense_outputs])

        # Compiling the RNN
        model.compile(optimizer=optimizer,loss=loss)
        return model
    
    elif n_layers == 6:
        # Inputs
        inputs= tf.keras.Input(shape=(window_size,n_cols))

        # GRU Layers
        l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_1_outputs = l_1(inputs)
        l_1_outputs = tf.keras.layers.Dropout(0.5)(l_1_outputs)

        l_2 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_2_outputs = l_2(l_1_outputs)
        l_2_outputs = tf.keras.layers.Dropout(0.5)(l_2_outputs)

        l_3 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_3_outputs = l_3(l_2_outputs)
        l_3_outputs = tf.keras.layers.Dropout(0.5)(l_3_outputs)

        l_4 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_4_outputs = l_4(l_3_outputs)
        l_4_outputs = tf.keras.layers.Dropout(0.5)(l_4_outputs)

        l_5 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
        l_5_outputs = l_5(l_4_outputs)
        l_5_outputs = tf.keras.layers.Dropout(0.5)(l_5_outputs)

        l_6 = tf.keras.layers.LSTM(int(n_neurons),activation=activation)
        l_6_outputs = l_6(l_5_outputs)
        l_6_outputs = tf.keras.layers.Dropout(0.5)(l_6_outputs)

        # Outputs
        dense = tf.keras.layers.Dense(4,activation=activation)
        dense_outputs = dense(l_6_outputs)

        model = tf.keras.models.Model([inputs],[dense_outputs])

        # Compiling the RNN
        model.compile(optimizer=optimizer,loss=loss)
        return model


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
# Data Prep for RNN
def rnn_data_prep(x,y,window_size,orig_data):
    # Reshaping the Data
    x = x.values
    y = y.values
    nrows = x.shape[0] - window_size + 1
    p,q = x.shape
    m,n = x.strides
    strided = np.lib.stride_tricks.as_strided

    x = strided(x,shape=(nrows,window_size,q),strides=(m,m,n))
    y = y[window_size-1:]
    orig_data = orig_data.loc[window_size-1:,:].reset_index(drop=True)
    
    return x,y,orig_data

#%%
# Features and Targets Scaling
def x_scaler(x_train,x_valid,x_test,scaler):

    x_train_reshaped = x_train.reshape(-1,x_train.shape[-1])
    x_valid_reshaped = x_valid.reshape(-1,x_valid.shape[-1])

    scaler.fit(x_train_reshaped)

    x_train_scaled = scaler.transform(x_train_reshaped)
    x_train_scaled = x_train_scaled.reshape(x_train.shape)

    x_valid_scaled = scaler.transform(x_valid_reshaped)
    x_valid_scaled = x_valid_scaled.reshape(x_valid.shape)

    x_test_scaled = []
    for i in range(len(x_test)):
        x_test_r = x_test[i].reshape(-1,x_test[i].shape[-1])
        x_test_s = scaler.transform(x_test_r)
        x_test_s = x_test_s.reshape(x_test[i].shape)
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
def get_folds(rnn_x,rnn_y,df,folds,selected_folds):
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

        X_train = rnn_x[train_idx,:,:]
        Y_train = rnn_y[train_idx,:]

        X_valid = rnn_x[valid_idx,:,:]
        Y_valid = rnn_y[valid_idx,:]

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