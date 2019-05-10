#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:59:24 2019

@author: zjy
"""

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

PAD_id = 0
EOS_id = 1
UNK_id = 2


def readlines(file):
    with open(file, 'r') as f:
        return [l.strip() for l in f.readlines()]


class Iterator(object):
    def __init__(self, files, vocab_tables, hparams):
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.output_buffer_size = hparams.batch_size * 1000

        # must assign one vocab_table to each file_tensor
        assert len(files) == len(vocab_tables)
        self.files = files
        self.vocab_tables = vocab_tables
        # create processor
        self.processor = Processor(hparams.num_buckets,
                                   self.output_buffer_size,
                                   hparams.num_parallel_calls,
                                   hparams.bucket_width,
                                   hparams.max_seq_len)

    def data_gen(self):  # Python generator
        f_pt = open(self.files[0], 'r', encoding='utf-8')
        g_pt = open(self.files[1], 'r', encoding='utf-8')
        if self.hparams.char:
            # example: 
            # 'abs <SEP> aa aa <SEP> bbb bb' to 'a b s <SEP> a a a a <SEP> b b b b b'
            def preprocess_fn(l):
                # remove SEP
                l = l.split(" <SEP> ")
                # remove space
                l = ["".join(ll.split()) for ll in l]
                l = [" ".join(list(ll)) for ll in l]
                return " <SEP> ".join(l)
        else:
            preprocess_fn = lambda l: l

        for x, y in zip(f_pt, g_pt):
            x, y = preprocess_fn(x.strip()), preprocess_fn(y.strip())
            yield x, " ".join([y, self.hparams.EOS])

    def build_dataset(self, train=True):  # Tensorflow generator
        """ iterator processing graph """
        dataSet = self.processor.gen_dataset(self.data_gen)
        dataSet = self.processor.zero_length(dataSet)
        # shuffle or not
        if train: dataSet = dataSet.shuffle(self.output_buffer_size, reshuffle_each_iteration=True)
        # string_split
        dataSet = self.processor.string_split(dataSet)
        # length_limit
        dataSet = self.processor.length_limit(dataSet)
        # look_up : transform words to ids
        dataSet = self.processor.look_up(dataSet, self.vocab_tables)
        # batching or bucketing
        dataSet = self.processor.batch_or_bucket(dataSet, self.batch_size, self.files, train)

        # parse to dictionary
        def parse_fn(*strings):
            keys = ['source', 'target']
            dict_element = {k: v for k, v in zip(keys, strings)}
            return dict_element, strings[-1]
        return dataSet.map(parse_fn)


# In[] data_processing
class Processor(object):
    """ a preprocessing pipeline """

    def __init__(self, num_buckets=10, output_buffer_size=0,
                 num_parallel_calls=10, bucket_width=4, max_seq_len=200):
        self.num_buckets = num_buckets
        self.output_buffer_size = output_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.max_seq_len = max_seq_len
        self.bucket_width = bucket_width

    def gen_dataset(self, gen):
        return tf.data.Dataset.from_generator(generator=gen,
                                              output_types=(tf.string, tf.string),
                                              output_shapes=(None, None))

    # string_split any number of string arguments
    def string_split(self, dataSet):
        string_split_op = lambda *strings: [tf.string_split([str]).values for str in strings]
        return dataSet.map(string_split_op, num_parallel_calls=self.num_parallel_calls).prefetch(
            self.output_buffer_size)

    # filter length()
    def zero_length(self, dataSet):
        length_limit_op = lambda *strings: tf.logical_and(tf.size(strings[0]) > 0, tf.size(strings[1]) > 0)
        return dataSet.filter(length_limit_op).prefetch(self.output_buffer_size)

    def length_limit(self, dataSet):
        length_limit_op = lambda *strings: [str[:self.max_seq_len] for str in strings]
        return dataSet.map(length_limit_op, num_parallel_calls=self.num_parallel_calls).prefetch(
            self.output_buffer_size)

    # look_up : words to ids
    def look_up(self, dataSet, vocab_tables):
        def look_up_op(*strings):
            return [tf.cast(vocab_table.lookup(str), tf.int32) for (vocab_table, str) in zip(vocab_tables, strings)]
        return dataSet.map(look_up_op, num_parallel_calls=self.num_parallel_calls).prefetch(self.output_buffer_size)

    # batch or bucket
    def batch_or_bucket(self, dataSet, batch_size, files, train=True):
        """ indicators : same length as strings, indicating which string to add length info (0 or 1)"""
        # padding_shapes and padding_values
        padded_shapes, padding_values = (), ()
        for _ in range(len(files)):
            padded_shapes += (tf.TensorShape([None]),)
            padding_values += (PAD_id,)

        # batch
        def batching_op(dataset):
            return dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        # bucket
        def bucket_op(left, right, *args):
            """ length_index : indicates which arg is used for bucketing """
            seq_len = tf.maximum(tf.size(left), tf.size(right))
            bucket_id = seq_len // self.bucket_width
            return tf.to_int64(tf.minimum(self.num_buckets, bucket_id))

        if self.num_buckets > 1:
            def reduce_func(unused_key, windowed_data):
                return batching_op(windowed_data)

            batched_dataset = dataSet.apply(tf.contrib.data.group_by_window(
                key_func=bucket_op, reduce_func=reduce_func, window_size=batch_size))
        else:
            batched_dataset = batching_op(dataSet)
        return batched_dataset


def gen_vocab_tables(hparams):  # vocab to word_id mapping
    # if hparams.use_pretrained_embedding:
    #     vocab_file = hparams.embedding_dir
    #     tf.logging.info("loading pretrained embedding from %s" % vocab_file)
    # else:
    vocab_file = hparams.vocab_file
    tf.logging.info("loading vocab from %s" % vocab_file)
    tf.logging.info("%d vocab loaded" % hparams.vocab_size)
    vocab_mapping_strings = tf.constant(hparams.vocab)
    vocab_table = lookup_ops.index_table_from_tensor(
        vocab_mapping_strings, default_value=UNK_id)
    return [vocab_table, vocab_table]


def train_input_fn(files, hparams):
    vocab_tables = gen_vocab_tables(hparams)
    iterator = Iterator(files, vocab_tables, hparams)
    return iterator.build_dataset(train=True)


def eval_input_fn(files, hparams):
    vocab_tables = gen_vocab_tables(hparams)
    """Load and return dataset of batched examples for use during evaluation."""
    iterator = Iterator(files, vocab_tables, hparams)
    return iterator.build_dataset(train=False)