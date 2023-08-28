import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tf.keras.utils.set_random_seed(0)

data_path = "M:/Dissertation/Data/"


#%%
# Time Series Scaler
class TSScaler(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(0,1))
    
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
# Attention Layer
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
        alpha = tf.keras.backend.expand_dims(alpha,axis=-1)
        context = x * alpha
        context = tf.keras.backend.sum(context,axis=1)
        return context
    
#%%
# RNN
def RNN(window_size,n_cols,n_neurons,activation,loss,optimizer):

    # Inputs
    inputs= tf.keras.Input(shape=(window_size,n_cols))

    # LSTM Layers
    l_1 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
    l_1_outputs = l_1(inputs)

    l_2 = tf.keras.layers.LSTM(int(n_neurons),activation=activation,return_sequences=True)
    l_2_outputs = l_2(l_1_outputs)

    l_3 = tf.keras.layers.LSTM(int(n_neurons/4),activation=activation,return_sequences=True)
    l_3_outputs = l_3(l_2_outputs)

    # Attention Layer
    attn = attention()
    attn_outputs = attn(l_3_outputs)

    # Outputs
    dense_1 = tf.keras.layers.Dense(1,activation=activation)
    dense_1_outputs = dense_1(attn_outputs)

    dense_2 = tf.keras.layers.Dense(1,activation=activation)
    dense_2_outputs = dense_2(attn_outputs)

    dense_3 = tf.keras.layers.Dense(1,activation=activation)
    dense_3_outputs = dense_3(attn_outputs)

    dense_4 = tf.keras.layers.Dense(1,activation=activation)
    dense_4_outputs = dense_4(attn_outputs)

    dense_5 = tf.keras.layers.Dense(1,activation=activation)
    dense_5_outputs = dense_5(attn_outputs)

    dense_6 = tf.keras.layers.Dense(1,activation=activation)
    dense_6_outputs = dense_6(attn_outputs)

    dense_7 = tf.keras.layers.Dense(1,activation=activation)
    dense_7_outputs = dense_7(attn_outputs)

    dense_8 = tf.keras.layers.Dense(1,activation=activation)
    dense_8_outputs = dense_8(attn_outputs)

    model = tf.keras.models.Model([inputs],[dense_1_outputs,dense_2_outputs,dense_3_outputs,dense_4_outputs,
                                            dense_5_outputs,dense_6_outputs,dense_7_outputs,dense_8_outputs])

    # Compiling the RNN
    model.compile(optimizer=optimizer,loss=loss)
    return model

#%%
# Scaling Function
def scale(df,train_index,test_index,scaler,save,exclude_cols,save_name):
    df_train = df.iloc[:train_index]
    df_valid = df.iloc[train_index:test_index]
    df_test = df.iloc[test_index:]
    

    if exclude_cols is not None:
        train_exc = df_train[exclude_cols].reset_index(drop=True)
        df_train = df_train.drop(exclude_cols,axis=1)

        valid_exc = df_valid[exclude_cols].reset_index(drop=True)
        df_valid= df_valid.drop(exclude_cols,axis=1)

        test_exc = df_test[exclude_cols].reset_index(drop=True)
        df_test = df_test.drop(exclude_cols,axis=1)
    
    scaler.fit(df_train)
    df_train_scaled = scaler.transform(df_train)
    df_train_scaled = pd.DataFrame(data=df_train_scaled,columns=df_train.columns)

    df_valid_scaled = scaler.transform(df_valid)
    df_valid_scaled = pd.DataFrame(data=df_valid_scaled,columns=df_valid.columns)

    df_test_scaled = scaler.transform(df_test)
    df_test_scaled = pd.DataFrame(data=df_test_scaled,columns=df_test.columns)

    if exclude_cols is not None:
        df_train_scaled = pd.concat([df_train_scaled,train_exc],axis=1)
        df_valid_scaled = pd.concat([df_valid_scaled,valid_exc],axis=1)
        df_test_scaled = pd.concat([df_test_scaled,test_exc],axis=1)
    
    if save:
        with open('scaler_'+save_name+'.pkl','wb') as file:
            pickle.dump(scaler,file)
    
    return df_train_scaled,df_valid_scaled,df_test_scaled


#%%
# Data Prepration for RNN
def rnn_data_prep(x,y,window_size,train_size,scaler,orig_data,save_name):
    if len(x)!=len(y):
        raise ValueError("length of x is not equal to y")
    
    orig_columns = orig_data.columns
    orig_data = orig_data.to_numpy()
    
    train_index = round(y.shape[0]*train_size)
    test_index = y.loc[y[y.columns[0]].isnull()].index[0]
    
    x_train_scaled,x_valid_scaled,x_test_scaled = scale(x,train_index,test_index,scaler,False,None,None)
    y_train_scaled,y_valid_scaled,y_test_scaled = scale(y,train_index,test_index,scaler,True,None,save_name)

    x_comb = pd.concat([x_train_scaled,x_valid_scaled,x_test_scaled]).reset_index(drop=True).to_numpy()
    y_comb = pd.concat([y_train_scaled,y_valid_scaled,y_test_scaled]).reset_index(drop=True).to_numpy()

    # Reshaping the Data
    nrows = x_comb.shape[0] - window_size + 1
    p,q = x_comb.shape
    m,n = x_comb.strides
    strided = np.lib.stride_tricks.as_strided

    x_comb = strided(x_comb,shape=(nrows,window_size,q),strides=(m,m,n))
    y_comb = y_comb[window_size-1:]
    orig_data = orig_data[window_size-1:]

    # Splitting the Data
    orig_test_x = orig_data[(test_index-window_size+1):,:]
    orig_test_x = pd.DataFrame(data=orig_test_x,columns=orig_columns)
    test_x = x_comb[(test_index-window_size+1):,:,:]

    orig_data = orig_data[:(test_index-window_size+1)]
    y_comb = y_comb[:(test_index-window_size+1)]
    x_comb = x_comb[:(test_index-window_size+1),:,:]

    orig_train_x = orig_data[:(train_index-window_size+1),:]
    orig_train_x = pd.DataFrame(data=orig_train_x,columns=orig_columns)
    train_x = x_comb[:(train_index-window_size+1),:,:]
    train_y = y_comb[:(train_index-window_size+1)]

    orig_valid_x = orig_data[(train_index-window_size+1):,:]
    orig_valid_x = pd.DataFrame(data=orig_valid_x,columns=orig_columns)
    valid_x = x_comb[(train_index-window_size+1):,:,:]
    valid_y = y_comb[(train_index-window_size+1):]
    
    return train_x,valid_x,test_x,orig_train_x,orig_valid_x,orig_test_x,train_y,valid_y

#%%
# Function for Compiling the Predictions
def compile_predictions(preds,n_vars,scaler_objs,col_names):
    if len(scaler_objs)!=n_vars:
        raise ValueError("Length Mismatch!")
    
    for i in range(n_vars):
        if i == 0:
            pred_df = pd.DataFrame(data=scaler_objs[i].inverse_transform(preds[i]),columns=[col_names[i]])
        else:
            pred_df = pd.concat([pred_df,pd.DataFrame(data=scaler_objs[i].inverse_transform(preds[i]),columns=[col_names[i]])],axis=1)
    return pred_df

#%%
# Custom Loss Function

def calculate_return(series):
    shifted_series = tf.roll(series,shift=-1,axis=0)
    return (shifted_series / series) - 1

def get_accuracy_score(y_true,y_pred):
    act_ret = calculate_return(y_true)
    pred_ret = calculate_return(y_pred)

    act_ret = tf.where(act_ret > 0, tf.ones_like(act_ret), -tf.ones_like(act_ret))
    pred_ret = tf.where(pred_ret > 0, tf.ones_like(pred_ret), -tf.ones_like(pred_ret))

    correct_predictions = tf.equal(act_ret,pred_ret)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))
    return accuracy

def weighted_mse(y_true,y_pred):
    loss = tf.reshape(tf.square(tf.math.log(y_true+1) - tf.math.log(y_pred+1)),(-1, 1))
    loss = loss + (1 - get_accuracy_score(y_true,y_pred))
    return tf.reduce_mean(loss,-1)