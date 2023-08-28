import pandas as pd
import numpy as np
import random
import os
import torch
import pickle
import tensorflow as tf
from accelerate import Accelerator
from torch.optim import AdamW
from evaluate import load
from typing import Optional, Iterable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from transformers import PretrainedConfig
from transformers import InformerConfig, InformerForPrediction
from gluonts.dataset.common import ListDataset
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform.sampler import InstanceSampler
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import TimeFeature
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (AddAgeFeature,AddObservedValuesIndicator,AddTimeFeatures,AsNumpyArray,Chain,ExpectedNumInstanceSampler,
                               InstanceSplitter,RemoveFields,SelectFields,SetField,TestSplitSampler,Transformation,ValidationSplitSampler,VstackFeatures,RenameFields)
tf.keras.utils.set_random_seed(0)


#%%
# Splitting Data into Train, Validation and Test sets based on the train and test size provided by the user

def data_split(df,train_size):
    
    train_index = round(df.shape[0]*train_size)
    test_index = df.loc[df.Portion=='Test'].index[0]

    df_train = df.iloc[:train_index]
    df_valid = df.iloc[train_index:test_index]
    df_test = df.iloc[test_index:]

    return df_train.drop('Portion',axis=1),df_valid.drop('Portion',axis=1),df_test.drop('Portion',axis=1)

#%%
# We will iterate over the individual time series of our dataset and add/remove fields or features

def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    # create list of fields to remove later
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            # AddAgeFeature(
            #     target_field=FieldName.TARGET,
            #     output_field=FieldName.FEAT_AGE,
            #     pred_length=config.prediction_length,
            #     log_scale=True,
            # ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            # VstackFeatures(
            #     output_field=FieldName.FEAT_TIME,
            #     input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
            #     + (
            #         [FieldName.FEAT_DYNAMIC_REAL]
            #         if config.num_dynamic_real_features > 0
            #         else []
            #     ),
            # ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )
#%%
# Next For training/validation/testing we create an InstanceSplitter which is used to 
# sample windows from the dataset (as, remember, we can't pass the entire history of values to the model due to time- and memory constraints).

# The instance splitter samples random context_length sized and subsequent prediction_length sized windows from the data, and appends 
# a past_ or future_ key to any temporal keys for the respective windows. This makes sure that the values will be split into past_values 
# and subsequent future_values keys, which will serve as the encoder and decoder inputs respectively. The same happens for any keys in the 
# time_series_fields argument:

def create_instance_splitter(config:PretrainedConfig,
                             mode:str,
                             train_sampler:Optional[InstanceSampler]=None,
                             validation_sampler: Optional[InstanceSampler] = None) -> Transformation:
    
    assert mode in ["train", "validation", "test"]

    instance_sampler = {"train": train_sampler or ExpectedNumInstanceSampler(num_instances=1.0, min_future=config.prediction_length),
                        "validation": validation_sampler or ValidationSplitSampler(min_future=config.prediction_length),
                        "test": TestSplitSampler()}[mode]

    return InstanceSplitter(target_field="values",is_pad_field=FieldName.IS_PAD,
                            start_field=FieldName.START,
                            forecast_start_field=FieldName.FORECAST_START,
                            instance_sampler=instance_sampler,
                            past_length=config.context_length + max(config.lags_sequence),
                            future_length=config.prediction_length,
                            time_series_fields=["time_features", "observed_mask"])
#%%
# DataLoaders allow us to have batches of (input, output) pairs - or in other words (past_values, future_values)

def create_train_dataloader(config: PretrainedConfig,
                            freq,
                            data,
                            batch_size:int,
                            num_batches_per_epoch:int,
                            shuffle_buffer_length:Optional[int]=None,
                            cache_data:bool=True,
                            **kwargs) -> Iterable:
    
    PREDICTION_INPUT_NAMES = ["past_time_features","past_values","past_observed_mask","future_time_features"]

    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")
    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + ["future_values","future_observed_mask"]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from all the possible transformed time series, 1 in our case)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream, is_train=True)
    
    return as_stacked_batches(training_instances,
                              batch_size=batch_size,
                              shuffle_buffer_length=shuffle_buffer_length,
                              field_names=TRAINING_INPUT_NAMES,
                              output_type=torch.tensor,
                              num_batches_per_epoch=num_batches_per_epoch)


def create_test_dataloader(config:PretrainedConfig,
                           freq,
                           data,
                           batch_size:int,
                           **kwargs):
    
    PREDICTION_INPUT_NAMES = ["past_time_features","past_values","past_observed_mask","future_time_features"]
   
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")
    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(testing_instances,
                              batch_size=batch_size,
                              output_type=torch.tensor,
                              field_names=PREDICTION_INPUT_NAMES)


def create_dataloader(data,num_of_variates,freq,config,type):
    data = pd.melt(data,id_vars=["start"],var_name="item_id",value_name="target")
    data = ListDataset(
        [
            {
                "item_id": idx,
                "target": data.loc[data.item_id==idx,'target'].values,
                "start": pd.Period(data['start'].iloc[0],freq='1H')
            } 
            for idx in list(data.item_id.unique())
        ],
        freq=freq,
    )

    if type == 'train':
        grouper = MultivariateGrouper(max_target_dim=num_of_variates)
        multi_variate_dataset = grouper(data)
        dataloader = create_train_dataloader(config=config,
                                                freq=freq,
                                                data=multi_variate_dataset,
                                                batch_size=256,
                                                num_batches_per_epoch=100,
                                                num_workers=2)
    else:
        grouper = MultivariateGrouper(max_target_dim=num_of_variates,num_test_dates=len(data)//num_of_variates)
        multi_variate_dataset = grouper(data)
        dataloader = create_test_dataloader(config=config,
                                            freq=freq,
                                            data=multi_variate_dataset,
                                            batch_size=32)
    return dataloader

#%%
# Train Model

def train_model(dataloader,model,device,optimizer,accelerator,epochs,config):
    loss_history = []
    model.to(device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()
    
    for epoch in range(epochs):
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(static_categorical_features=batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
                            static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
                            past_time_features=batch["past_time_features"].to(device),
                            past_values=batch["past_values"].to(device),
                            future_time_features=batch["future_time_features"].to(device),
                            future_values=batch["future_values"].to(device),
                            past_observed_mask=batch["past_observed_mask"].to(device),
                            future_observed_mask=batch["future_observed_mask"].to(device))
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())
            # if idx % 100 == 0:
            #     print(f"Epoch {epoch}, Loss = ",loss.item())
    return model

#%%
# Generate Forecasts

def generate_forecasts(data,model,num_of_variates,context_length,lags,freq,config,device):
    forecasts = np.array([],dtype=np.float32).reshape(0,num_of_variates)
    for k,v in data.iterrows():

        if k>=(context_length+max(lags)-1):

            model.eval()
            forecasts_ = []
            temp = data.loc[k-(context_length+max(lags)-1):k,:]
            dataloader = create_dataloader(temp,num_of_variates,freq,config,'test')

            for batch in dataloader:
                outputs = model.generate(static_categorical_features = batch["static_categorical_features"].to(device) if config.num_static_categorical_features > 0 else None,
                                        static_real_features=batch["static_real_features"].to(device) if config.num_static_real_features > 0 else None,
                                        past_time_features=batch["past_time_features"].to(device),
                                        past_values=batch["past_values"].to(device),
                                        future_time_features=batch["future_time_features"].to(device),
                                        past_observed_mask=batch["past_observed_mask"].to(device))
                forecasts_.append(outputs.sequences.cpu().numpy())

            forecasts = np.concatenate([np.median(np.vstack(forecasts_),1).squeeze(0),forecasts],axis=0)
    return forecasts