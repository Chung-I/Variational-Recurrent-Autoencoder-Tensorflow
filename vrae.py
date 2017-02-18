# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time
import logging
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import seq2seq_model
from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_string("model_dir", "models", "directory of the model.")
tf.app.flags.DEFINE_boolean("new", True, "whether this is a new model or not.")
tf.app.flags.DEFINE_string("do", "train", "what to do. accepts train, interpolate, sample, and decode.")
tf.app.flags.DEFINE_string("input", None, "input filename for reconstruct sample, and interpolate.")
tf.app.flags.DEFINE_string("output", None, "output filename for reconstruct sample, and interpolate.")

FLAGS = tf.app.flags.FLAGS

def prelu(x):
  with tf.variable_scope("prelu") as scope:
    alphas = tf.get_variable("alphas", [], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    return tf.nn.relu(x) - tf.mul(alphas, tf.nn.relu(-x))


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


def read_data(source_path, target_path, config, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(config.buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < config.buckets[n][0] and
      len(target) < config.buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in config.buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(config.buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, config, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float32
  optimizer = None
  if not forward_only:
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
  if config.activation == "elu":
    activation = tf.nn.elu
  elif config.activation == "prelu":
    activation = prelu
  else:
    activation = tf.identity

  weight_initializer = tf.orthogonal_initializer if config.orthogonal_initializer else tf.uniform_unit_scaling_initializer
  bias_initializer = tf.zeros_initializer

  model = seq2seq_model.Seq2SeqModel(
      config.en_vocab_size,
      config.fr_vocab_size,
      config.buckets,
      config.size,
      config.num_layers,
      config.latent_dim,
      config.max_gradient_norm,
      config.batch_size,
      config.learning_rate,
      config.kl_min,
      config.word_dropout_keep_prob,
      config.anneal,
      config.use_lstm,
      optimizer=optimizer,
      activation=activation,
      forward_only=forward_only,
      feed_previous=config.feed_previous,
      bidirectional=config.bidirectional,
      weight_initializer=weight_initializer,
      bias_initializer=bias_initializer,
      iaf=config.iaf,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if not FLAGS.new and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train(config):
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing WMT data in %s" % config.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
      config.data_dir, config.en_vocab_size, config.fr_vocab_size, config.load_embeddings)

  with tf.Session() as sess:
    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)

    # Create model.
    print("Creating %d layers of %d units." % (config.num_layers, config.size))
    model = create_model(sess, config, False)

    if not config.probabilistic:
      self.kl_rate_update(0.0)

    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir,"train"), graph=sess.graph)
    dev_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "test"), graph=sess.graph)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % config.max_train_data_size)

    dev_set = read_data(en_dev, fr_dev, config)
    train_set = read_data(en_train, fr_train, config, config.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(config.buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    KL_loss = 0.0
    current_step = model.global_step.eval()
    step_loss_summaries = []
    step_KL_loss_summaries = []
    overall_start_time = time.time()
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, step_KL_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False, config.probabilistic)

      if config.anneal and model.global_step.eval() > config.kl_rate_rise_time and model.kl_rate < 1:
        new_kl_rate = model.kl_rate.eval() + config.kl_rate_rise_factor
        sess.run(model.kl_rate_update, feed_dict={'new_kl_rate': new_kl_rate})

      step_time += (time.time() - start_time) / config.steps_per_checkpoint
      step_loss_summaries.append(tf.Summary(value=[tf.Summary.Value(tag="step loss", simple_value=float(step_loss))]))
      step_KL_loss_summaries.append(tf.Summary(value=[tf.Summary.Value(tag="KL step loss", simple_value=float(step_KL_loss))]))
      loss += step_loss / config.steps_per_checkpoint
      KL_loss += step_KL_loss / config.steps_per_checkpoint
      current_step = model.global_step.eval()

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % config.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        print ("global step %d learning rate %.4f step-time %.2f KL divergence "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, KL_loss))
        wall_time = time.time() - overall_start_time
        print("time passed: {0}".format(wall_time))

        # Add perplexity, KL divergence to summary and stats.
        perp_summary = tf.Summary(value=[tf.Summary.Value(tag="train perplexity", simple_value=perplexity)])
        train_writer.add_summary(perp_summary, current_step)
        KL_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="KL divergence", simple_value=KL_loss)])
        train_writer.add_summary(KL_loss_summary, current_step)
        for i, summary in enumerate(step_loss_summaries):
          train_writer.add_summary(summary, current_step - 200 + i)
        step_loss_summaries = []
        for i, summary in enumerate(step_KL_loss_summaries):
          train_writer.add_summary(summary, current_step - 200 + i)
        step_KL_loss_summaries = []

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name + ".ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss, KL_loss = 0.0, 0.0, 0.0

        # Run evals on development set and print their perplexity.
        eval_losses = []
        eval_KL_losses = []
        eval_bucket_num = 0
        for bucket_id in xrange(len(config.buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          eval_bucket_num += 1
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, eval_KL_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True, config.probabilistic)
          eval_losses.append(float(eval_loss))
          eval_KL_losses.append(float(eval_KL_loss))
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

          eval_perp_summary = tf.Summary(value=[tf.Summary.Value(tag="eval perplexity for bucket {0}".format(bucket_id), simple_value=eval_ppx)])
          dev_writer.add_summary(eval_perp_summary, current_step)

        mean_eval_loss = sum(eval_losses) / float(eval_bucket_num)
        mean_eval_KL_loss = sum(eval_KL_losses) / float(eval_bucket_num)
        mean_eval_ppx = math.exp(float(mean_eval_loss))
        print("  eval: mean perplexity {0}".format(mean_eval_ppx))

        eval_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean eval loss", simple_value=float(mean_eval_ppx))])
        dev_writer.add_summary(eval_loss_summary, current_step)
        eval_KL_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean eval loss", simple_value=float(mean_eval_KL_loss))])
        dev_writer.add_summary(eval_KL_loss_summary, current_step)


def reconstruct(sess, model, config):
  model.batch_size = 1  # We decode one sentence at a time.
  model.probabilistic = config.probabilistic
  beam_size = config.beam_size

  # Load vocabularies.
  en_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.in" % config.en_vocab_size)
  fr_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.out" % config.fr_vocab_size)
  en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
  _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

  # Decode from standard input.
  outputs = []
  with gfile.GFile(FLAGS.input, "r") as fs:
    sentences = fs.readlines()
  for i, sentence in  enumerate(sentences):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)
    # Which bucket does it belong to?
    bucket_id = len(config.buckets) - 1
    for i, bucket in enumerate(config.buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    else:
      logging.warning("Sentence truncated: %s", sentence) 

    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)

    if beam_size > 1:
      path, symbol, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
              target_weights, bucket_id, True, config.probabilistic, beam_size)

      k = output_logits[0]
      paths = []
      for kk in range(beam_size):
        paths.append([])
      curr = range(beam_size)
      num_steps = len(path)
      for i in range(num_steps-1, -1, -1):
        for kk in range(beam_size):
          paths[kk].append(symbol[i][curr[kk]])
          curr[kk] = path[i][curr[kk]]
      recos = set()
      for kk in range(beam_size):
        output = [int(logit)  for logit in paths[kk][::-1]]

        if EOS_ID in output:
          output = output[:output.index(EOS_ID)]
        output = " ".join([rev_fr_vocab[word] for word in output]) + "\n"
        outputs.append(output)

    else:
    # Get output logits for the sentence.
      _, _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True, config.probabilistic)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      output = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in output:
        output = output[:output.index(data_utils.EOS_ID)]
      output = " ".join([rev_fr_vocab[word] for word in output]) + "\n"
      outputs.append(output)
  with gfile.GFile(FLAGS.output, "w") as enc_dec_f:
    for output in outputs:
      enc_dec_f.write(output)


def encode(sess, model, config, sentences):
  # Load vocabularies.
  en_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.in" % config.en_vocab_size)
  fr_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.out" % config.fr_vocab_size)
  en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
  _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
  
  means = []
  logvars = []
  for i, sentence in enumerate(sentences):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)
    # Which bucket does it belong to?
    bucket_id = len(config.buckets) - 1
    for i, bucket in enumerate(config.buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    else:
      logging.warning("Sentence truncated: %s", sentence) 

        # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, _, _ = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    mean, logvar = model.encode_to_latent(sess, encoder_inputs, bucket_id)
    means.append(mean)
    logvars.append(logvar)

  return means, logvars


def decode(sess, model, config, means, logvars, bucket_id):
  fr_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.out" % config.fr_vocab_size)
  _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

  _, decoder_inputs, target_weights = model.get_batch(
      {bucket_id: [([], [])]}, bucket_id)
  outputs = []
  for mean, logvar in zip(means, logvars):
    mean = mean.reshape(1,-1)
    logvar = logvar.reshape(1,-1)
    output_logits = model.decode_from_latent(sess, mean, logvar, bucket_id, decoder_inputs, target_weights)
    output = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in output:
      output = output[:output.index(data_utils.EOS_ID)]
    output = " ".join([rev_fr_vocab[word] for word in output]) + "\n"
    outputs.append(output)

  return outputs
  # Print out French sentence corresponding to outputs.

def n_sample(sess, model, config):
  bucket_id = len(config.buckets) - 1
  with gfile.GFile(FLAGS.input, "r") as fs:
    sentences = fs.readlines()
  mean, logvar = encode(sess, model, config, sentences)
  mean = mean[0][0]
  logvar = logvar[0][0]
  means = [mean] * config.num_pts
  neg_inf_logvar = np.full(logvar.shape, -800.0, dtype=np.float32)
  logvars = [neg_inf_logvar] + [logvar] * (config.num_pts - 1)
  outputs = decode(sess, model, config, means, logvars, bucket_id)
  with gfile.GFile(FLAGS.output, "w") as sample_f:
    for output in outputs:
      sample_f.write(output)
  

def interpolate(sess, model, config, means, logvars, num_pts):
  if len(means) != 2:
    raise ValueError("there should be two sentences when interpolating."
                     "number of setences: %d." % len(means))
  if num_pts < 3:
    raise ValueError("there should be more than two points when interpolating."
                     "number of points: %d." % num_pts)
  pts = []
  for s, e in zip(means[0][0].tolist(),means[1][0].tolist()):
    pts.append(np.linspace(s, e, num_pts))

  pts = np.array(pts)
  pts = pts.T
  pts = [np.array(pt) for pt in pts.tolist()]
  bucket_id = len(config.buckets) - 1
  logvars = [np.full(pt.shape, -800.0, dtype=np.float32) for pt in pts]
  outputs = decode(sess, model, config, pts, logvars, bucket_id)

  return outputs

def encode_interpolate(sess, model, config):
  with gfile.GFile(FLAGS.input, "r") as fs:
    sentences = fs.readlines()
  model.batch_size = 1
  model.probabilistic = config.probabilistic
  means, logvars = encode(sess, model, config, sentences)
  outputs = interpolate(sess, model, config, means, logvars, config.num_pts)
  with gfile.GFile(FLAGS.output, "w") as interp_f:
    for output in outputs:
      interp_f.write(output)

class Struct(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)
    if not self.__dict__.get("kl_min"):
      self.__dict__.update({ "kl_min": None })
    if not self.__dict__.get("max_gradient_norm"):
      self.__dict__.update({ "max_gradient_norm": 5.0 })
    if not self.__dict__.get("load_embeddings"):
      self.__dict__.update({ "load_embeddings": False })
    if not self.__dict__.get("batch_size"):
      self.__dict__.update({ "batch_size": 1 })
    if not self.__dict__.get("learning_rate"):
      self.__dict__.update({ "learning_rate": 0.001 })
    if not self.__dict__.get("anneal"):
      self.__dict__.update({ "anneal": False })
    if not self.__dict__.get("beam_size"):
      self.__dict__.update({ "beam_size": 1 })
    if self.__dict__.get("beam_size") > 1:
      raise NotImplementedError("Beam search is still under implementation.")
  def update(self, **entries):
    self.__dict__.update(entries)


def main(_):

  with open(os.path.join(FLAGS.model_dir, "config.json")) as config_file:
    configs = json.load(config_file)

  FLAGS.model_name = os.path.basename(os.path.normpath(FLAGS.model_dir)) 
  behavior = ["train", "interpolate", "reconstruct", "sample"]
  if FLAGS.do not in behavior:
    raise ValueError("argument \"do\" is not one of the following: train, interpolate, decode or sample.")

  if FLAGS.do != "train":
    FLAGS.new = False

  config = Struct(**configs["model"])
  config.update(**configs[FLAGS.do])
  interp_config = Struct(**configs["model"])
  interp_config.update(**configs["interpolate"])
  enc_dec_config = Struct(**configs["model"])
  enc_dec_config.update(**configs["reconstruct"])
  sample_config = Struct(**configs["model"])
  sample_config.update(**configs["sample"])

  if FLAGS.do == "reconstruct":
    with tf.Session() as sess:
      model = create_model(sess, enc_dec_config, True)
      reconstruct(sess, model, enc_dec_config)
  elif FLAGS.do == "interpolate":
    with tf.Session() as sess:
      model = create_model(sess, interp_config, True)
      encode_interpolate(sess, model, interp_config)
  elif FLAGS.do == "sample":
    with tf.Session() as sess:
      model = create_model(sess, sample_config, True)
      n_sample(sess, model, config)
  elif FLAGS.do == "train":
    train(config)

if __name__ == "__main__":
  tf.app.run()
