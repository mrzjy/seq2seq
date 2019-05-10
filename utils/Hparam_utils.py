import tensorflow as tf
import argparse
import os
import numpy as np


# hard coded, do not change this
PAD = '<PAD>'
PAD_id = 0
EOS = '<EOS>'
EOS_id = 1
UNK = '<UNK>'
UNK_id = 2
SEP = '<SEP>'
SEP_id = 3


def create_standard_hparams():
    return tf.contrib.training.HParams(
        # envirionment
        num_gpus=0,

        # Data constraints
        num_buckets=20,
        bucket_width=5,
        max_seq_len=200,
        num_parallel_calls=8,

        # Data format
        PAD=PAD,
        PAD_id=PAD_id,
        EOS=EOS,
        EOS_id=EOS_id,
        UNK=UNK,
        UNK_id=UNK_id,
        SEP=SEP,
        SEP_id=SEP_id,

        # dir
        data_dir='data',
        test_data_dir='',
        model_dir='saved_model',
        export_dir='exported_model',
        embedding_dir='embedding/pca_embed_100',

        # data
        train_x='train_x',
        train_y='train_y',
        valid_x='test_x',
        valid_y='test_y',
        test_x='test_x',
        test_y='test_y',
        pred_x='pred_x',  # only pred_x is absolute path, for others we'll join the path with data_dir

        # vocab
        vocab_dir='',
        vocab_file='vocab',

        # Networks
        model="rnn",  # rnn, rnn_attention, trans
        hidden_size=512,
        # RNN
        # ======================================================================
        embed_size=256,  # For Transformer, embed_size is FORCED to equal to hidden size
        # Transformer
        # ======================================================================
        initializer_gain=1.0,  # Used in trainable variable initialization.
        num_hidden_layers=3,  # Number of layers in the encoder and decoder stacks.
        num_heads=4,  # Number of heads to use in multi-headed attention.
        filter_size=2048,  # Inner layer dimension in the feedforward network (about 4 times the hidden size).
        allow_ffn_pad=True,
        label_smoothing=0.1,
        extra_decode_length=40,

        # ======================================================================
        # Default prediction params
        max_decode_length=50,
        beam_size=10,
        length_penalty_weight=0.8,
        coverage_penalty_weight=0.0,
        # TODO
        MMI_decode=1,  # use MMI when decoding(inference), applicable only for predict.py

        # tricks
        char=True,  # char-level tokenization
        reverse_sequence=True,   # reverse sequence generation (will be reversed back to normal finally)

        # Train
        dropout=0.15,
        threshold=0.5,
        summary_frequency=80000,
        log_frequency=200,
        eval_frequency=100000,
        batch_size=256,
        total_num_epochs=20,
        learning_rate=2e-4,
        optimizer="adam",
        max_gradient_norm=5.0,

        # infer
        infer_mode="beam_search",
        get_all_beams=True  # only used when running predict.py
    )


def create_hparams():
    hparams = create_standard_hparams()
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--hparams', type=str, default="",
                        help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()
    hparams.parse(args.hparams)

    if hparams.test_data_dir == '':
        hparams.set_hparam('test_data_dir', hparams.data_dir)

    hparams.set_hparam("train_x", os.path.join(hparams.data_dir, hparams.train_x))
    hparams.set_hparam("train_y", os.path.join(hparams.data_dir, hparams.train_y))
    hparams.set_hparam("valid_x", os.path.join(hparams.data_dir, hparams.valid_x))
    hparams.set_hparam("valid_y", os.path.join(hparams.data_dir, hparams.valid_y))
    hparams.set_hparam("test_x", os.path.join(hparams.test_data_dir, hparams.test_x))
    hparams.set_hparam("test_y", os.path.join(hparams.test_data_dir, hparams.test_y))

    # if hparams.use_pretrained_embedding:
    #     embeddings = load_embedding(hparams)
    #     hparams.add_hparam("embeddings", embeddings)
    #     hparams.set_hparam("src_embedding_size", hparams.pretrained_embedding_size)
    #     hparams.set_hparam("tgt_embedding_size", hparams.pretrained_embedding_size)
    # else:
    if hparams.vocab_dir == '':
        hparams.set_hparam("vocab_dir", hparams.data_dir)
        tf.logging.info("Vocab_dir not defined, using same dir as data_dir: %s" % hparams.data_dir)
    hparams.set_hparam("vocab_file", os.path.join(hparams.vocab_dir, hparams.vocab_file))
    # set vocab
    if hparams.char:
        hparams.vocab_file += "_char"
        hparams.model_dir += "_char"
        tf.logging.info("Using char-level vocab setting, vocab_file name must end with \"_char\" ")

    with open(hparams.vocab_file, 'r') as f:
        vocab = [l.strip() for l in f.readlines()]
        hparams.add_hparam("vocab", vocab)
        hparams.add_hparam("vocab_size", len(vocab))
        tf.logging.info("Found %d vocab_size from %s" % (len(vocab), hparams.vocab_file))

    # print mission
    print_mission(hparams)
    return hparams


def print_mission(hparams):
    tf.logging.info("Training schedule:")
    tf.logging.info("\t1. Train for {} epochs (Train-Eval-inferenceTest) in total.".format(hparams.total_num_epochs))
    tf.logging.info("\t2. Evaluate every %d steps." % hparams.eval_frequency)
    tf.logging.info("\t3. Batch_size=%d." % hparams.batch_size)


def load_embedding(hparams):
    with open(hparams.embedding_dir, 'r', encoding='utf-8') as f:
        lines = [l.strip().split() for l in f.readlines()]
    embedding = np.array([[float(i) for i in l[1:]] for i, l in enumerate(lines)])
    return embedding
