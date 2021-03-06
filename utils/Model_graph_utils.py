#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:43:40 2018

@author: zjy
"""
import tensorflow as tf
import math

_NEG_INF = -1e9
eps = 1e-07


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

  Args:
    x: int tensor with any shape
    padding_value: int value that

  Returns:
    flaot tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.

  Args:
    x: int tensor with shape [batch_size, length]

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def record_scalars(metric_dict):
    for key, value in metric_dict.items():
        tf.contrib.summary.scalar(name=key, tensor=value)


def get_seq_length(seq_ids):
    padding = tf.to_float(tf.equal(seq_ids, 0))
    pad_len = tf.cast(tf.reduce_sum(padding, axis=1), dtype=tf.int32)
    seq_len = tf.shape(seq_ids)[1] - pad_len
    return seq_len


def optimization(learning_rate, total_loss, tvars, params):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-08, )
    gradients = tf.gradients(total_loss, tvars)
    clipped_grads, grad_norm = tf.clip_by_global_norm(gradients, params["max_gradient_norm"])
    return optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=tf.train.get_global_step())


def get_distribution_strategy(distribution_strategy="default", num_gpus=0,
                              num_workers=1, all_reduce_alg=None):
    if num_gpus <= 1:
        return None

    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == "off":
        if num_gpus > 1 or num_workers > 1:
            raise ValueError(
                "When {} GPUs and  {} workers are specified, distribution_strategy "
                "flag cannot be set to 'off'.".format(num_gpus, num_workers))
        return None

    if distribution_strategy in ("mirrored", "default"):
        if num_gpus == 0:
            devices = ["device:CPU:0"]
        else:
            devices = ["device:GPU:%d" % i for i in range(num_gpus)]
        if all_reduce_alg:
            return tf.contrib.distribute.MirroredStrategy(
                devices=devices,
                cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
                    all_reduce_alg, num_packs=2))
        else:
            return tf.contrib.distribute.MirroredStrategy(devices=devices)

    if distribution_strategy == "parameter_server":
        return tf.contrib.distribute.experimental.ParameterServerStrategy()

    raise ValueError(
        "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def print_params(params):
    print("# Trainable variables")
    total_num_params = 0
    for param in params:
        num_params = 1
        for dim in param.get_shape():
            num_params *= dim.value
        total_num_params += num_params
        print("    %s, shape : %s, Number of params : %s" % (
            param.name, str(param.get_shape()), num_params))
    print("# Total trainable parameters :", total_num_params)
