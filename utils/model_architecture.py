import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '-1'

import random
import tensorflow as tf
import numpy as np
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import keras
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Add,
    TimeDistributed, LayerNormalization
)
from tcn import TCN
from utils.custom_layers import SqueezeLastDim, VectorQuantizer


def build_model(
    input_shape,
    PAD_VALUE,
    num_vq_embeddings=64,
    vq_beta=0.25
):
    inputs = Input(shape=input_shape)

    # Mask padded values
    x = layers.Masking(mask_value=PAD_VALUE)(inputs)

    # TCN backbone
    tcn_out = TCN(
        nb_filters=64,
        kernel_size=4,
        nb_stacks=2,
        dilations=[1, 2, 4, 8, 16],
        padding="causal",
        return_sequences=True,
        use_skip_connections=True,
        activation="elu"
    )(x)
    tcn_out = LayerNormalization()(tcn_out)

    # LSTM branch
    lstm_out = LSTM(64, return_sequences=True)(tcn_out)
    lstm_out = Dropout(0.2, seed=42)(lstm_out)
    lstm_out = LSTM(64, return_sequences=True)(lstm_out)
    lstm_out = Dropout(0.2, seed=42)(lstm_out)

    # Residual connection
    x = Add()([lstm_out, tcn_out])

    # Direct Dense branch
    x_direct = TimeDistributed(Dense(128, activation="elu"))(x)
    out_direct = TimeDistributed(Dense(1, activation="linear"))(x_direct)
    out_direct = SqueezeLastDim()(out_direct)

    # VQ branch
    x_vq = VectorQuantizer(
        num_embeddings=num_vq_embeddings,
        embedding_dim=x.shape[-1],
        beta=vq_beta
    )(x)
    x_vq = LayerNormalization()(x_vq)
    x_vq = TimeDistributed(Dense(128, activation="elu"))(x_vq)
    out_vq = TimeDistributed(Dense(1, activation="linear"))(x_vq)
    out_vq = SqueezeLastDim()(out_vq)

    # Average both branches
    outputs = layers.Average()([out_direct, out_vq])

    model = Model(inputs, outputs)
    return model



def compile_model(model):
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae']
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    return model, early_stopping, reduce_lr
    