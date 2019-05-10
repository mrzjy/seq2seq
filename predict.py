#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:29 2019

@author: zjy
"""

import tensorflow as tf
from utils.Hparam_utils import create_hparams
from utils.Common_utils import Monitor, Tokenizer, pred_input_fn_builder
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PredictMonitor(Monitor):
    def __init__(self, hparams):
        # init misc variables
        self.init_misc(hparams)

        # prepare estimator
        self.hparams = hparams
        self.Model = self.choose_model()
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_file)
        self.estimator_config = self.config_estimator_session()
        self.estimator_model_dir = self.config_estimator_model_dir()
        self.estimator_model_fn = self.config_estimator_model_fn()
        self.estimator = self.create_estimator()

    def create_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.estimator_model_fn,
            model_dir=self.estimator_model_dir,
            params=self.hparams.__dict__,
            config=None)

    def ids_to_words(self, res):
        if self.hparams.get_all_beams:
            # [beam, batch, length], [beam, batch]
            beam_ids, beam_length = res["predicts"], res["predicts_len"]
            beam_wise_result = []
            for beam in range(beam_ids.shape[0]):
                beam_wise_result.append(self.tokenizer.ids_2_words(beam_ids[beam, :, :], beam_length[beam, :], EOS_end=True))
            return beam_wise_result
        else:
            return self.tokenizer.ids_2_words(res["predicts"], res["predicts_len"], EOS_end=True)

    def run(self):
        if self.hparams.get_all_beams:
            predictions = [[] for _ in range(hparams.beam_size)]
            
        else:
            predictions = []
        tf.logging.info("Sample generation...")
        pred_fn = pred_input_fn_builder(self.hparams.pred_x, self.tokenizer, self.hparams, max_seq_len=40)
        for i, res in enumerate(self.estimator.predict(pred_fn, yield_single_examples=False)):
            if self.hparams.get_all_beams:
                for pred, beam_res in zip(predictions, self.ids_to_words(res)):
                    pred.extend(beam_res)
            else:
                predictions.extend(self.ids_to_words(res))
            print(i+1, "batches predicted")
            if i > 5:
                break
        return predictions


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # hparams
    hparams = create_hparams()
    # monitor
    monitor = PredictMonitor(hparams)
    # list of predictions
    predictions = monitor.run()

    if hparams.get_all_beams:
        for beam_id, res in enumerate(predictions):
            print("beam", beam_id)
            for r in res:
                print(r)
            print()
    else:
        with open("predictions.txt", "w+") as f:
            print("\n".join(predictions), file=f)
