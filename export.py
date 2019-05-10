#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

@author: zjy
"""

import tensorflow as tf
from utils.Hparam_utils import create_hparams
from utils.Common_utils import Monitor
from utils.Data_utils import gen_vocab_tables
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ExportMonitor(Monitor):
    def __init__(self, hparams):
        # prepare estimator
        self.hparams = hparams
        self.Model = self.choose_model()
        self.estimator_model_dir = self.config_estimator_model_dir()
        self.estimator_model_fn = self.config_estimator_model_fn()
        self.estimator = self.create_estimator()

    def create_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.estimator_model_fn,
            model_dir=self.estimator_model_dir,
            params=self.hparams.__dict__,
            config=None)


def inference_input_receiver_fn():
    vocab_table = gen_vocab_tables(hparams)[0]
    max_seq_len = 200
    source_raw = tf.placeholder(dtype=tf.string, shape=(None, None), name='source')
    receiver_tensors = {"source": source_raw}
    features = {"source": preprocess(source_raw, vocab_table, max_seq_len)}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def preprocess(source, vocab_table, max_seq_len):
    # max length
    source = source[:, :max_seq_len]
    # vocab
    source = tf.cast(vocab_table.lookup(source), tf.int32)
    return source


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # hparams
    hparams = create_hparams()
    # monitor
    monitor = ExportMonitor(hparams)
    # export estimator
    export_dir_base = os.path.join(hparams.export_dir, hparams.model)
    monitor.estimator.export_savedmodel(export_dir_base, inference_input_receiver_fn)