#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:24:32 2019

@author: zjy
"""
import tensorflow as tf

from utils.Data_utils import train_input_fn, eval_input_fn
from utils import Eval_utils
from utils.Model_graph_utils import get_seq_length, get_distribution_strategy
from utils.Model_graph_utils import print_params, optimization

import os
import glob
import shutil
import random

tf.logging.set_verbosity(tf.logging.INFO)


class Monitor:
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

        # prepare data
        self.data_files = self.prepare_data_files()

    def choose_model(self):
        if self.hparams.model.lower().startswith("trans"):
            from utils.Model_Transformer_utils import Transformer
            return Transformer
        elif self.hparams.model.lower() == "rnn":
            from utils.Model_RNN_utils import Seq2Seq
            return Seq2Seq
        else:
            from utils.Model_RNN_Attention_utils import Seq2Seq_Attn
            return Seq2Seq_Attn

    def config_estimator_model_fn(self):
        """Defines how to train, evaluate and predict from the transformer model."""

        def model_fn(features, labels, mode, params):
            inputs, inputs_len = features['source'], get_seq_length(features['source'])
            # Create model and get output logits.
            model = self.Model(params, mode)
            # prediction
            if mode == tf.estimator.ModeKeys.PREDICT:
                predicts, predicts_len = model(inputs)

                if params["reverse_sequence"]:  # remember to leave EOS
                    if params["get_all_beams"]:
                        predicts = tf.map_fn(lambda x: tf.reverse_sequence(x[0], x[1] - 1, seq_axis=1),
                                             [predicts, predicts_len],  # beam-major, so perform map_fn beam-wisely
                                             dtype=tf.int32)
                    else:
                        predicts = tf.reverse_sequence(predicts, predicts_len - 1, seq_axis=1)
                # prediction
                return tf.estimator.EstimatorSpec(
                    tf.estimator.ModeKeys.PREDICT,
                    predictions={"inputs": inputs, "inputs_len": inputs_len,
                                 "predicts": predicts, "predicts_len": predicts_len})
            else:
                targets, targets_len = features['target'], get_seq_length(features['target'])
                if params["reverse_sequence"]:  # remember to leave EOS
                    targets = tf.reverse_sequence(targets, targets_len - 1, seq_axis=1)
                logits, output_len = model(inputs, targets)
                # train
                if self.show_model_graph_when_1st_run:
                    print_params(tf.trainable_variables())
                    self.show_model_graph_when_1st_run = False
                # Calculate model loss.
                loss, _ = Eval_utils.padded_cross_entropy_loss(
                    logits, targets, params["label_smoothing"], params["vocab_size"])
                num_samples = tf.cast(tf.reduce_sum(targets_len), tf.float32)
                loss = tf.reduce_sum(loss) / num_samples

                # eval
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(
                        mode=mode, loss=loss, predictions={"predictions": logits},
                        eval_metric_ops=Eval_utils.get_eval_metrics(logits, labels, params))

                train_op = optimization(params["learning_rate"], loss, tf.trainable_variables(), params)
                predicts, predicts_len = tf.argmax(logits, axis=-1), output_len
                if params["reverse_sequence"]:
                    targets = tf.reverse_sequence(targets, targets_len - 1, seq_axis=1)
                    predicts = tf.reverse_sequence(predicts, predicts_len - 1, seq_axis=1)
                trainHook = TrainHook(inputs, inputs_len,
                                      targets, targets_len,
                                      predicts, predicts_len,
                                      self.tokenizer, params)
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                                  train_op=train_op,
                                                  training_hooks=[trainHook])

        return model_fn

    def run(self):
        best_result = 0
        # data files
        train_files, valid_files, test_files = self.data_files
        # Loop training/evaluation cycles
        num_total_loops = max(1, self.total_num_steps // self.hparams.eval_frequency)
        for i in range(num_total_loops):
            tf.logging.info("Starting %d/%d train-valid-test process" % (i + 1, num_total_loops))
            # Train the model for single_iteration_train_steps or until the input fn
            # runs out of examples (if single_iteration_train_steps is None).
            self.estimator.train(lambda: train_input_fn(train_files, self.hparams),
                                 steps=self.hparams.eval_frequency)
            # evaluate
            results = self.estimator.evaluate(lambda: eval_input_fn(valid_files, self.hparams))
            metric = results['metrics/approx_bleu_score']
            # keep best model according to metric
            best_result = keep_best_model(self.estimator, best_result, metric)
            # show one inference sample
            tf.logging.info("")
            tf.logging.info("Sample generation...")
            pred_fn = pred_input_fn_builder(self.hparams.test_x, self.tokenizer, self.hparams, max_seq_len=40)
            for res in self.estimator.predict(pred_fn, yield_single_examples=False):
                inputs = self.tokenizer.ids_2_words(res["inputs"], res["inputs_len"], EOS_end=False)
                preds = self.tokenizer.ids_2_words(res["predicts"], res["predicts_len"], EOS_end=True)
                rand_id = random.randint(0, len(inputs) - 1)
                tf.logging.info("   Input: %s" % inputs[rand_id])
                tf.logging.info("Response: %s" % preds[rand_id])
                tf.logging.info("")
                break

    def init_misc(self, hparams):
        self.show_model_graph_when_1st_run = True

    def config_estimator_session(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        # distributed training
        distribution_strategy = get_distribution_strategy(
            distribution_strategy="default",
            num_gpus=self.hparams.num_gpus)
        session_config = tf.estimator.RunConfig(
            session_config=gpu_config,
            save_summary_steps=self.hparams.summary_frequency,
            log_step_count_steps=self.hparams.log_frequency,
            train_distribute=distribution_strategy if self.hparams.num_gpus else None)
        return session_config

    def config_estimator_model_dir(self):
        base_dir = "saved_model"
        if self.hparams.model.lower().startswith("trans"):
            return base_dir + "/Transformer"
        elif self.hparams.model.lower() == "rnn":
            return base_dir + "/S2S"
        else:  # rnn attention
            return base_dir + "/S2SA"

    def create_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.estimator_model_fn,
            model_dir=self.estimator_model_dir,
            params=self.hparams.__dict__,
            config=self.estimator_config)

    def prepare_data_files(self):
        num_samples = 0
        train_files = [self.hparams.train_x, self.hparams.train_y]
        valid_files = [self.hparams.valid_x, self.hparams.valid_y]
        tf.logging.info("Estimating total_num_steps (reading %s)" % self.hparams.train_y)
        with open(self.hparams.train_y, 'r') as f:  # add random filter for large files
            for i, l in enumerate(f):
                num_samples += 1
        self.total_num_steps = estimate_total_steps(num_samples, self.hparams)
        test_files = [self.hparams.test_x, self.hparams.test_y]
        return train_files, valid_files, test_files


class TrainHook(tf.train.SessionRunHook):
    def __init__(self, inputs, inputs_len, targets, targets_len, predicts, predicts_len, tokenizer, params):
        super(TrainHook, self).__init__()
        self.predicts_len = predicts_len
        self.predicts = predicts
        self.targets_len = targets_len
        self.targets = targets
        self.inputs_len = inputs_len
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.params = params

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs({"source": [self.inputs, self.inputs_len],
                                        "target": [self.targets, self.targets_len],
                                        "predict": [self.predicts, self.predicts_len]})

    def after_run(self, run_context, run_values):
        if self._step % self.params["log_frequency"] == 0:
            res = run_values.results
            src = self.tokenizer.ids_2_words(res["source"][0], res["source"][1], EOS_end=False)
            preds = self.tokenizer.ids_2_words(res["predict"][0], res["predict"][1], EOS_end=True)
            tgt = self.tokenizer.ids_2_words(res["target"][0], res["target"][1], EOS_end=True)
            rand_id = random.randint(0, len(src) - 1)
            tf.logging.info("   Input: %s" % src[rand_id])
            tf.logging.info("  Target: %s" % tgt[rand_id])
            tf.logging.info(" Predict: %s" % preds[rand_id])
            tf.logging.info("")


def pred_input_fn_builder(input_file, tokenizer, hparams, max_seq_len=40):
    UNK_id = hparams.UNK_id
    if hparams.char:
        # example: 
        # 'abs <SEP> aa aa <SEP> bbb bb' to 'a b s <SEP> a a a a <SEP> b b b b b'
        def preprocess_fn(l):
            # remove SEP
            l = l.split(" <SEP> ")
            # remove space
            l = ["".join(ll.split()) for ll in l]
            l = [" ".join(list(ll)) for ll in l]
            return " <SEP> ".join(l).split()
    else:
        preprocess_fn = lambda l: l.split()

    def gen():
        seq_ids = []
        with open(input_file, 'r') as f:
            textlines = [l.strip() for l in f.readlines()]
        for line in textlines:
            words = preprocess_fn(line)
            seq_id = [tokenizer.w2i.get(word, UNK_id) for word in words[-max_seq_len:]]
            # zero-pad
            while len(seq_id) < max_seq_len:
                seq_id.append(0)
            seq_ids.append(seq_id)

            if len(seq_ids) == hparams.batch_size:
                yield {'source': seq_ids}
                seq_ids = []
        # last time
        yield {'source': seq_ids}

    def input_fn():
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types={'source': tf.int32},
            output_shapes={'source': (None, max_seq_len)})
        return dataset

    return input_fn


class Tokenizer:
    def __init__(self, tool='hanlp', vocab_path='data/vocab'):
        self.tool_name = tool
        if tool.lower() == 'jieba':
            import jieba
            self.tool = jieba
        elif tool.lower() == 'hanlp':
            from pyhanlp import HanLP
            self.tool = HanLP
        else:
            raise Exception("Unknown tokenization tool")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [l.strip().split()[0] for l in f.readlines()]

        self.w2i = {word: i for i, word in enumerate(self.vocab)}
        self.i2w = {i: word for i, word in enumerate(self.vocab)}

    def special_tokenize(self, string):
        substring_list = string.split(" <SEP> ")
        reunited = " <SEP> ".join([self.tokenize_line(sub) for sub in substring_list])
        return reunited.split()

    def tokenize_line(self, string, return_string=True):
        if self.tool_name.lower() == 'jieba':
            tokenized = [w for w in self.tool.cut(string) if w != ' ']
        elif self.tool_name.lower() == 'hanlp':
            tokenized = [term.word for term in self.tool.segment(string) if term != ' ']
        else:
            raise Exception("Unknown tokenization tool")
        if len(tokenized) == 0:
            return string
        return " ".join(tokenized) if return_string else tokenized

    def ids_2_words(self, output_ids, output_lens, EOS_end=False):
        sentences = []
        assert output_ids.shape[0] == output_lens.shape[0]
        for out_id, out_len in zip(output_ids, output_lens):
            if EOS_end:
                out_len -= 1  # exclude the EOS token
            word_list = [self.i2w[word_id] for word_id in out_id[:out_len]]
            sentences.append(" ".join(word_list))
        return sentences


def estimate_total_steps(num_samples, hparams):
    total_num_steps_per_epoch = num_samples // hparams.batch_size
    if hparams.num_gpus:
        total_num_steps_per_epoch = total_num_steps_per_epoch // hparams.num_gpus
    tf.logging.info("Estimated total_num_steps = %d for %d epochs" % (
        total_num_steps_per_epoch * hparams.total_num_epochs,
        hparams.total_num_epochs))
    return total_num_steps_per_epoch * hparams.total_num_epochs


def keep_best_model(estimator, best_result, new_eval_result):
    if new_eval_result > best_result:
        BEST_CHECKPOINTS_PATH = "best/"
        tf.logging.info(
            'Saving a new better model ({:.3f} better than {:.3f})...'.format(new_eval_result, best_result))
        # copy the checkpoints files *.meta *.index, *.data* each time there is a better result, no cleanup for max
        # amount of files here
        latest_checkpoint = estimator.latest_checkpoint()
        for name in glob.glob(latest_checkpoint + '.*'):
            copy_to = os.path.join(os.path.dirname(latest_checkpoint), BEST_CHECKPOINTS_PATH)
            if not os.path.exists(copy_to):
                os.makedirs(copy_to)
            shutil.copy(name, os.path.join(copy_to, os.path.basename(name)))
        # also save the text file used by the estimator api to find the best checkpoint
        with open(os.path.join(copy_to, "checkpoint"), 'w+') as f:
            f.write("model_checkpoint_path: \"{}\"".format(os.path.basename(latest_checkpoint)))
        best_result = new_eval_result
    return best_result
