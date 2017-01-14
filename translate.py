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
import random
import sys
import time
import logging
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model
import h5py
from tensorflow.python.platform import gfile
from tensorflow.contrib.tensorboard.plugins import projector
from utils.adamax import AdamaxOptimizer
import utils.prelu
import pdb

tf.app.flags.DEFINE_string("model_dir", "input.txt", "directory of the model.")
tf.app.flags.DEFINE_boolean("new", True, "whether this is a new model or not.")
tf.app.flags.DEFINE_string("do", "train", "what to do. accepts train, interpolate, sample, and decode.")

FLAGS = tf.app.flags.FLAGS


def maybe_create_statistics(config):
  stat_file_name = "stats/" + FLAGS.model_name + ".json" 
  if FLAGS.new:
    if os.path.exists(stat_file_name):
      print("error: create an already existed statistics file")
      sys.exit()
    stats = {}
    stats['hyperparameters'] = config.__dict__
    stats['model_name'] = FLAGS.model_dir
    stats['train_perplexity'] = {}
    stats['train_KL_divergence'] = {}
    stats['eval_KL_divergence'] = {}
    stats['eval_perplexity'] = {}
    stats['wall_time'] = {}
    with open(stat_file_name, "w") as statfile:
      statfile.write(json.dumps(stats))
  else:
    with open(stat_file_name, "r") as statfile:
      statjson = statfile.read()
      stats = json.loads(statjson)
      hparams = stats['hyperparameters']
  return stats


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
  optimizer = AdamaxOptimizer(config.learning_rate) if config.adamax else tf.train.AdamOptimizer(config.learning_rate)  #adamax currently not supported
  if config.elu: #this is deprecated and will soon be removed
    activation = tf.nn.elu
  elif config.activation == "elu":
    activation = tf.nn.elu
  elif config.activation == "prelu":
    activation = utils.prelu.prelu
  elif config.activation == "none":
    activation = tf.identity
  else:
    activation = tf.nn.relu
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
      config.latent_splits,
      config.Lambda,
      config.word_dropout_keep_prob,
      config.beam_size,
      config.annealing,
      config.lower_bound_KL,
      config.kl_rate_rise_time,
      config.kl_rate_rise_factor,
      config.use_lstm,
      config.mean_logvar_split,
      config.load_embeddings,
      config.Lambda_annealing,
      optimizer=optimizer,
      activation=activation,
      dnn_in_between=config.dnn_in_between,
      probabilistic=config.probabilistic,
      batch_norm=config.batch_norm,
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


def train(config, encode_decode_config, interp_config):
  """Train a en->fr translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing WMT data in %s" % config.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
      config.data_dir, config.en_vocab_size, config.fr_vocab_size, config.load_embeddings)

  stats = maybe_create_statistics(config)

  with tf.Session() as sess:
    if not os.path.exists(FLAGS.model_dir):
      os.makedirs(FLAGS.model_dir)


    stat_file_name = "stats/" + FLAGS.model_name + ".json" 
    # Create model.
    print("Creating %d layers of %d units." % (config.num_layers, config.size))
    model = create_model(sess, config, False)

    train_writer = tf.summary.FileWriter(FLAGS.model_dir+ "/train", graph=sess.graph)
    dev_writer = tf.summary.FileWriter(FLAGS.model_dir + "/test", graph=sess.graph)

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
    if config.load_embeddings:
      with h5py.File(config.data_dir + "/vocab{0}".format(config.en_vocab_size) + '.en.embeddings.h5','r') as h5f:
        enc_embeddings = h5f['embeddings'][:]
      sess.run(model.enc_embedding_init_op, feed_dict={model.enc_embedding_placeholder: enc_embeddings})
      del enc_embeddings
      with h5py.File(config.data_dir + "/vocab{0}".format(config.fr_vocab_size) + '.fr.embeddings.h5','r') as h5f:
        dec_embeddings = h5f['embeddings'][:]
      sess.run(model.dec_embedding_init_op, feed_dict={model.dec_embedding_placeholder: dec_embeddings})
      del dec_embeddings

    projector_config = projector.ProjectorConfig()
    vis_enc_embedding = projector_config.embeddings.add()
    vis_dec_embedding = projector_config.embeddings.add()
    vis_enc_embedding.tensor_name = model.enc_embedding.name
    vis_dec_embedding.tensor_name = model.dec_embedding.name
    vis_enc_embedding.metadata_path = os.path.join("/data/home/iLikeNLP/zh_translate/",
            config.data_dir, 'enc_embedding{0}.tsv'.format(config.en_vocab_size))
    vis_dec_embedding.metadata_path = os.path.join("/data/home/iLikeNLP/zh_translate/",
            config.data_dir, 'dec_embedding{0}.tsv'.format(config.fr_vocab_size))
    projector.visualize_embeddings(train_writer, projector_config)

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
                                   target_weights, bucket_id, False)
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
        stats['wall_time'][str(current_step)] = wall_time
        if config.Lambda_annealing:
          if perplexity < 1.05:
            model.probabilistic = True
            if model.Lambda.eval() > 16:
              sess.run(model.Lambda_divide_by_two_op)
          print("lambda: {0}".format(model.Lambda.eval()))

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


        stats['train_perplexity'][str(current_step)] = perplexity
        stats['train_KL_divergence'][str(current_step)] = KL_loss

        if config.annealing:
          if current_step >= config.kl_rate_rise_time and model.kl_rate.eval() < 1:
            print("current kl rate: {0}".format(model.kl_rate.eval()))
            sess.run(model.kl_rate_rise_op)

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
                                       target_weights, bucket_id, True)
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

        stats['eval_perplexity'][str(current_step)] = mean_eval_ppx
        stats['eval_KL_divergence'][str(current_step)] = mean_eval_KL_loss
        eval_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean eval loss", simple_value=float(mean_eval_ppx))])
        dev_writer.add_summary(eval_loss_summary, current_step)
        eval_KL_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="mean eval loss", simple_value=float(mean_eval_KL_loss))])
        dev_writer.add_summary(eval_KL_loss_summary, current_step)

        with open(stat_file_name, "w") as statfile:
          statfile.write(json.dumps(stats))

        outputs = encode_interpolate(sess, model, interp_config)
        with gfile.GFile(FLAGS.model_dir + "/{0}.{1}.interpolate.txt".format(FLAGS.model_name,current_step), "w") as interp_file:
          for output in outputs:
            interp_file.write(output)

        outputs = encode_decode(sess, model, encode_decode_config)
        with gfile.GFile(FLAGS.model_dir + "/{0}.{1}.encode_decode.txt".format(FLAGS.model_name,current_step), "w") as enc_dec_file:
          for output in outputs:
            enc_dec_file.write(output)

        model.probabilistic = config.probabilistic
        model.batch_size = config.batch_size


def encode_decode(sess, model, config):
  model.batch_size = 1  # We decode one sentence at a time.
  model.probabilistic = config.probabilistic

  # Load vocabularies.
  en_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.en" % config.en_vocab_size)
  fr_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.fr" % config.fr_vocab_size)
  en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
  _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

  # Decode from standard input.
  outputs = []
  with gfile.GFile(config.input_file, "r") as fs:
    sentences = fs.readlines()
  with gfile.GFile(FLAGS.model_dir + ".output.txt", "w") as fo:
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

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      output = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in output:
        output = output[:output.index(data_utils.EOS_ID)]
      output = " ".join([rev_fr_vocab[word] for word in output]) + "\n"
      outputs.append(output)
  return outputs


def encode(sess, model, config, sentences):
  # Load vocabularies.
  en_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.en" % config.en_vocab_size)
  fr_vocab_path = os.path.join(config.data_dir,
                               "vocab%d.fr" % config.fr_vocab_size)
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
                               "vocab%d.fr" % config.fr_vocab_size)
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

def n_sample(sess, model, sentence, num_sample):
  mean, logvar = encode(sess, model, [sentence])
  mean = mean[0][0][0]
  logvar = logvar[0][0][0]
  means = [mean] * num_sample
  zero_logvar = np.zeros(shape=logvar.shape)
  logvars = [zero_logvar] + [logvar] * (num_sample - 1)
  outputs = decode(sess, model, means, logvars, len(config.buckets) - 1)
  with gfile.GFile(FLAGS.model_dir + ".{0}_sample.txt".format(num_sample), "w") as fo:
    for output in outputs:
      fo.write(output)
  

def interpolate(sess, model, config, means, logvars, num_pts):
  if len(means) != 2:
    raise ValueError("there should be two sentences when interpolating."
                     "number of setences: %d." % len(means))
  if num_pts < 3:
    raise ValueError("there should be more than two points when interpolating."
                     "number of points: %d." % num_pts)
  pts = []
  for s, e in zip(means[0][0][0].tolist(),means[1][0][0].tolist()):
    pts.append(np.linspace(s, e, num_pts))


  pts = np.array(pts)
  pts = pts.T
  pts = [np.array(pt) for pt in pts.tolist()]
  bucket_id = len(config.buckets) - 1
  logvars = [np.zeros(shape=pt.shape) for pt in pts]
  outputs = decode(sess, model, config, pts, logvars, bucket_id)

  return outputs

def encode_interpolate(sess, model, config):
  with gfile.GFile(config.input_file, "r") as fs:
    sentences = fs.readlines()
  model.batch_size = 1
  model.probabilistic = config.probabilistic
  means, logvars = encode(sess, model, config, sentences)
  outputs = interpolate(sess, model, config, means, logvars, config.num_pts)
  return outputs

class Struct(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)
    if not self.__dict__.get('iaf'):
      self.__dict__.update({ "iaf": False })
    if not self.__dict__.get('adamax'):
      self.__dict__.update({ "adamax": False })
    if not self.__dict__.get('elu'):
      self.__dict__.update({ "elu": False })


def main(_):
  
  with open(os.path.join(FLAGS.model_dir, "config.json")) as config_file:
    configs = json.load(config_file)

  FLAGS.model_name = os.path.basename(os.path.normpath(FLAGS.model_dir)) 
  behavior = ["train", "interpolate", "encode_decode", "sample"]
  if FLAGS.do not in behavior:
    raise ValueError("argument \"do\" must be one of the following: train, interpolate, decode or sample.")

  config = configs[FLAGS.do]
  config = Struct(**config)
  interp_config = Struct(**configs["interpolate"])
  encode_decode_config = Struct(**configs["encode_decode"])
  if FLAGS.do == "encode_decode":
    with tf.Session() as sess:
      model = create_model(sess, encode_decode_config, True)
      outputs = encode_decode(sess, model, encode_decode_config)
    with gfile.GFile(os.path.join(FLAGS.model_dir, "encode_decode.txt"), "w") as enc_dec_f:
      for output in outputs:
        enc_dec_f.write(output)
  elif FLAGS.do == "interpolate":
    with tf.Session() as sess:
      model = create_model(sess, interp_config, True)
      outputs = encode_interpolate(sess, model, interp_config)
    with gfile.GFile(os.path.join(FLAGS.model_dir, "interpolate.txt"), "w") as interp_f:
      for output in outputs:
        interp_f.write(output)
  elif FLAGS.do == "train":
    train(config, encode_decode_config, interp_config)

if __name__ == "__main__":
  tf.app.run()
