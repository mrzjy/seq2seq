#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:34:23 2019

@author: zjy
"""
import tensorflow as tf
from utils.Model_RNN_utils import Seq2Seq, Embedding, Encoder, Decoder, Output_layer


class Seq2Seq_Attn(Seq2Seq):
    def __init__(self, params, mode):
        """
        Initialize layers to build.
        """
        self.mode = mode
        self.params = params
        self.scope_name = "S2SA"

        # embedding
        self.embedding = Embedding(vocab_size=params["vocab_size"],
                                   embed_size=params["embed_size"])
        # seq2seq framework
        self.encoder = Encoder(params, mode)
        if mode == tf.contrib.learn.ModeKeys.INFER:
            self.decoder = AttentiveInferenceDecoder(params, mode)
        else:
            self.decoder = AttentiveDecoder(params, mode)
        self.output_layer = Output_layer(params, self.embedding.linear)  # tied weights


class AttentiveDecoder(Decoder):
    def _maybe_dropout(self, tensor):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            return tf.nn.dropout(tensor, 1.0 - self.dropout)
        else:
            return tensor

    def call(self, source_encoding, source_length=None, source_state=None,
             embedding_matrix=None, output_layer=None, target_embed=None, target_length=None):
        assert self.mode != tf.contrib.learn.ModeKeys.INFER
        # attention
        attention_keys = self._maybe_dropout(source_encoding)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.hidden_size, attention_keys, memory_sequence_length=source_length)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell, attention_mechanism, attention_layer_size=self.hidden_size)
        batch_size = tf.shape(source_encoding)[0]
        attn_initial_state = attn_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=source_state)

        # shift targets
        if target_embed is not None and target_length is not None:
            # Shift targets to the right, and remove the last element (hence the same length as before)
            with tf.name_scope("shift_targets"):
                decoder_inputs_emb = tf.pad(target_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_emb, target_length)
        else:
            raise Exception("Train or Eval mode must provide target_embed and target_length")

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attn_cell,
            helper=helper,
            initial_state=attn_initial_state,
            output_layer=output_layer)

        # Dynamic decoding
        outputs, final_state, outputs_length = tf.contrib.seq2seq.dynamic_decode(decoder)
        return outputs.rnn_output, outputs_length


class AttentiveInferenceDecoder(AttentiveDecoder):
    def call(self, source_encoding, source_length=None, source_state=None,
             embedding_matrix=None, output_layer=None, target_embed=None, target_length=None):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        # attention
        attention_keys = source_encoding
        if self.infer_mode == "beam_search":
            tf.logging.info("Inference using beam search:width=%d,alpha=%.2f,coverage_penalty=%.2f" % (
               self.beam_width, self.length_penalty_weight, self.coverage_penalty_weight ))
            tiled_attention_keys = tf.contrib.seq2seq.tile_batch(attention_keys, multiplier=self.beam_width)
            tiled_attention_length = tf.contrib.seq2seq.tile_batch(source_length, multiplier=self.beam_width)
            tiled_encoder_state = tf.contrib.seq2seq.tile_batch(source_state, multiplier=self.beam_width)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.hidden_size, tiled_attention_keys, memory_sequence_length=tiled_attention_length)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell, attention_mechanism, attention_layer_size=self.hidden_size)
        else:
            tf.logging.info("Inference using greedy inference")
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.hidden_size, attention_keys, memory_sequence_length=source_length)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell, attention_mechanism, attention_layer_size=self.hidden_size)

        batch_size = tf.shape(source_encoding)[0]
        if self.infer_mode == "beam_search":
            decoder_initial_state = attn_cell.zero_state(batch_size * self.beam_width, tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_state)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=attn_cell,
                embedding=embedding_matrix,
                start_tokens=tf.fill([batch_size], 0),
                end_token=1,
                initial_state=decoder_initial_state,
                beam_width=self.beam_width,
                output_layer=output_layer,
                length_penalty_weight=self.length_penalty_weight,
                coverage_penalty_weight=self.coverage_penalty_weight)
        else:
            decoder_initial_state = attn_cell.zero_state(batch_size, tf.float32).clone(cell_state=source_state)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=embedding_matrix,
                start_tokens=tf.fill([batch_size], 0),
                end_token=1)  # EOS_id is hardcoded as 1
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=output_layer)

        outputs, final_state, outputs_length = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.max_decode_length)

        if self.infer_mode == "beam_search":
            if self.get_all_beams:  # all to beam-major
                # outputs.predicted_ids is [batch, length, beam]
                predicted_ids = tf.transpose(outputs.predicted_ids, [2, 0, 1])
                # outputs_length is [batch, beam]
                outputs_length = tf.transpose(outputs_length, [1, 0])
            else:
                predicted_ids = outputs.predicted_ids[:, :, 0]
                outputs_length = outputs_length[:, 0]
            return predicted_ids, outputs_length
        else:
            return outputs.sample_id, outputs_length