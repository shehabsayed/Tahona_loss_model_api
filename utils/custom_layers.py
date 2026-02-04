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

from tensorflow.keras import layers


# Squeezes the last dimension (used for model output shaping)
class SqueezeLastDim(layers.Layer):
    def call(self, x):
        return tf.squeeze(x, axis=-1)

    def get_config(self):
        config = super().get_config()
        return config


# Vector Quantization layer (VQ-VAE style)
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = self.add_weight(
            name="embeddings_vqvae",
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=tf.random_uniform_initializer(seed=42),
            trainable=True
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean(
            (quantized - tf.stop_gradient(x)) ** 2
        )

        self.add_loss(self.beta * commitment_loss + codebook_loss)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        return tf.argmin(distances, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "beta": self.beta
        })
        return config
