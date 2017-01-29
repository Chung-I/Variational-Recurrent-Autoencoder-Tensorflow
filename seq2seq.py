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

"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np
from utils.distributions import DiagonalGaussian

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access



def prelu(_x):
  with tf.variable_scope("prelu"):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
      initializer=tf.constant_initializer(0.0),
      dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, word_dropout_keep_prob=1, replace_inp=None,
                loop_function=None, scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    seq_len = len(decoder_inputs)
    keep = tf.select(tf.random_uniform([seq_len]) < word_dropout_keep_prob,
            tf.fill([seq_len], True), tf.fill([seq_len], False))
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          if word_dropout_keep_prob < 1:
            inp = tf.cond(keep[i], lambda: loop_function(prev, i), lambda: replace_inp)
          else:
            inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def beam_rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None,output_projection=None, beam_size=1):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    log_beam_probs, beam_path, beam_symbols = [],[],[]
    state_size = int(initial_state.get_shape().with_rank(2)[1])

    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i,log_beam_probs, beam_path, beam_symbols)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      input_size = inp.get_shape().with_rank(2)[1]
      x = inp
      output, state = cell(x, state)

      if loop_function is not None:
        prev = output
      if  i ==0:
          states =[]
          for kk in range(beam_size):
                states.append(state)
          state = tf.reshape(tf.concat(0, states), [-1, state_size])

      outputs.append(tf.argmax(nn_ops.xw_plus_b(
          output, output_projection[0], output_projection[1]), dimension=1))
  return outputs, state, tf.reshape(tf.concat(0, beam_path),[-1,beam_size]), tf.reshape(tf.concat(0, beam_symbols),[-1,beam_size])


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          embedding,
                          num_symbols,
                          embedding_size,
                          word_dropout_keep_prob=1,
                          replace_input=None,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          weight_initializer=None,
                          beam_size=1,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    if not embedding:
      embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size],
              initializer=weight_initializer())

    if beam_size > 1:
        loop_function = _extract_beam_search(
        embedding, beam_size,num_symbols,embedding_size,  output_projection,
        update_embedding_for_previous)
    else:
        loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None

    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    if beam_size > 1:
        return beam_rnn_decoder(emb_inp, initial_state, cell,loop_function=loop_function,
                output_projection=output_projection, beam_size=beam_size)

    return rnn_decoder(emb_inp, initial_state, cell, word_dropout_keep_prob, replace_input,
                       loop_function=loop_function)


def embedding_attention_encoder(encoder_inputs,
                                cell,
                                num_encoder_symbols,
                                embedding_size,
                                dtype=None,
                                scope=None):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_encoder", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    return encoder_state, attention_states


def embedding_encoder(encoder_inputs,
                      cell,
                      embedding,
                      num_symbols,
                      embedding_size,
                      bidirectional=False,
                      dtype=None,
                      weight_initializer=None,
                      scope=None):

  with variable_scope.variable_scope(
      scope or "embedding_encoder", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    if not embedding:
      embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size],
              initializer=weight_initializer())
    emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in encoder_inputs]
    if bidirectional:
      _, output_state_fw, output_state_bw = rnn.bidirectional_rnn(cell, cell, emb_inp,
              dtype=dtype)
      encoder_state = tf.concat(1, [output_state_fw, output_state_bw])
    else:
      _, encoder_state = rnn.rnn(
        cell, emb_inp, dtype=dtype)

    return encoder_state


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses


def autoencoder_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, encoder, decoder, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        encoder_state = encoder(encoder_inputs[:bucket[0]])
        bucket_outputs, _ = decoder(encoder_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses


def sample(means,
           logvars,
           latent_dim,
           iaf=True,
           kl_min=None,
           anneal=False,
           kl_rate=None,
           dtype=None):
  """Perform sampling and calculate KL divergence.

  Args:
    means: tensor of shape (batch_size, latent_dim)
    logvars: tensor of shape (batch_size, latent_dim)
    latent_dim: dimension of latent space.
    iaf: perform linear IAF or not.
    kl_min: lower bound for KL divergence.
    anneal: perform KL cost annealing or not.
    kl_rate: KL divergence is multiplied by kl_rate if anneal is set to True.
  Returns:
    latent_vector: latent variable after sampling. A vector of shape (batch_size, latent_dim).
    kl_obj: objective to be minimized for the KL term.
    kl_cost: real KL divergence.
  """
  if iaf:
    with tf.variable_scope('iaf'):
      prior = DiagonalGaussian(tf.zeros_like(means, dtype=dtype),
              tf.zeros_like(logvars, dtype=dtype))
      posterior = DiagonalGaussian(means, logvars)
      z = posterior.sample

      logqs = posterior.logps(z)
      L = tf.get_variable("inverse_cholesky", [latent_dim, latent_dim], dtype=dtype, initializer=tf.zeros_initializer)
      diag_one = tf.ones([latent_dim], dtype=dtype)
      L = tf.matrix_set_diag(L, diag_one)
      mask = np.tril(np.ones([latent_dim,latent_dim]))
      L = L * mask
      latent_vector = tf.matmul(z, L)
      logps = prior.logps(latent_vector)
      kl_cost = logqs - logps
  else:
    noise = tf.random_normal(tf.shape(mean))
    sample = mean + tf.exp(0.5 * logvar) * noise
    kl_cost = -0.5 * (logvars - tf.square(means) -
        tf.exp(logvars) + 1.0)
  kl_ave = tf.reduce_mean(kl_cost, [0]) #mean of kl_cost over batches
  kl_obj = kl_cost = tf.reduce_sum(kl_ave)
  if kl_min:
    kl_obj = tf.reduce_sum(tf.maximum(kl_ave, kl_min))
  if anneal:
    kl_obj = kl_obj * kl_rate

  return latent_vector, kl_obj, kl_cost #both kl_obj and kl_cost are scalar


def encoder_to_latent(encoder_state,
                      embedding_size,
                      latent_dim,
                      num_layers,
                      activation=tf.nn.relu,
                      use_lstm=False,
                      enc_state_bidirectional=False,
                      dtype=None):
  concat_state_size = num_layers * embedding_size
  if enc_state_bidirectional:
    concat_state_size *= 2
  if use_lstm:
    concat_state_size *= 2
    if num_layers > 1:
      encoder_state = list(map(lambda state_tuple: tf.concat(1, state_tuple), encoder_state))
    else:
      encoder_state = tf.concat(1, encoder_state)
  if num_layers > 1:
    encoder_state = tf.concat(1, encoder_state)
  with tf.variable_scope('encoder_to_latent'):
    w = tf.get_variable("w",[concat_state_size, 2 * latent_dim],
      dtype=dtype)
    b = tf.get_variable("b", [2 * latent_dim], dtype=dtype)
    mean_logvar = prelu(tf.matmul(encoder_state, w) + b)
    mean, logvar = tf.split(1, 2, mean_logvar)

  return mean, logvar


def latent_to_decoder(latent_vector,
                      embedding_size,
                      latent_dim,
                      num_layers,
                      activation=tf.nn.relu,
                      use_lstm=False,
                      dtype=None):

  concat_state_size = num_layers * embedding_size
  if use_lstm:
    concat_state_size *= 2
  with tf.variable_scope('latent_to_decoder'):
    w = tf.get_variable("w",[latent_dim, concat_state_size],
      dtype=dtype)
    b = tf.get_variable("b", [concat_state_size], dtype=dtype)
    decoder_initial_state = prelu(tf.matmul(latent_vector, w) + b)
  if num_layers > 1:
    decoder_initial_state = tuple(tf.split(1, num_layers, decoder_initial_state))
    if use_lstm:
      decoder_initial_state = [tuple(tf.split(1, 2, single_layer_state)) for single_layer_state in decoder_initial_state]
  elif use_lstm:
    decoder_initial_state = tuple(tf.split(1, 2, decoder_initial_state))

  return decoder_initial_state


def variational_autoencoder_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, encoder, decoder, enc_latent, latent_dec, sample, kl_f,
                       probabilistic=False,
                       softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  KL_divergences = []
  with ops.name_scope(name, "variational_autoencoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        encoder_last_state = encoder(encoder_inputs[:bucket[0]])
        mean, logvar = enc_latent(encoder_last_state)
        if probabilistic:
          latent_vector = sample(mean, logvar)
        else:
          latent_vector = mean
        decoder_initial_state = latent_dec(latent_vector)
        bucket_outputs, _ = decoder(decoder_initial_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        total_size = math_ops.add_n(weights[:bucket[1]])
        total_size += 1e-12
        KL_divergences.append(tf.reduce_mean(kl_f(mean, logvar) / total_size))
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses, KL_divergences


def variational_encoder_with_buckets(encoder_inputs, buckets, encoder,
                       enc_latent, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))

  all_inputs = encoder_inputs
  means = []
  logvars = []
  with ops.name_scope(name, "variational_encoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        encoder_last_state = encoder(encoder_inputs[:bucket[0]])
        mean, logvar = enc_latent(encoder_last_state)
        means.append(mean)
        logvars.append(logvar)

  return means, logvars


def variational_decoder_with_buckets(means, logvars, decoder_inputs,
                       targets, weights,
                       buckets, decoder, latent_dec, sample,
                       softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = decoder_inputs + targets + weights
  losses = []
  outputs = []
  KL_objs = []
  KL_costs = []
  with ops.name_scope(name, "variational_decoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True if j > 0 else None):

        latent_vector, kl_obj, kl_cost = sample(means[j], logvars[j])
        decoder_initial_state = latent_dec(latent_vector)

        bucket_outputs, _ = decoder(decoder_initial_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        total_size = math_ops.add_n(weights[:bucket[1]])
        total_size += 1e-12
        KL_objs.append(tf.reduce_mean(kl_obj / total_size))
        KL_costs.append(tf.reduce_mean(kl_cost / total_size))
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses, KL_objs, KL_costs


def variational_beam_decoder_with_buckets(means, logvars, decoder_inputs,
                       targets, weights,
                       buckets, decoder, latent_dec, kl_f, sample, iaf=False,
                       softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = decoder_inputs + targets + weights
  losses = []
  outputs = []
  beam_paths = []
  beam_path = []
  KL_divergences = []
  with ops.name_scope(name, "variational_decoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True if j > 0 else None):
        latent_vector, kl_cost = sample(means[j], logvars[j])
        decoder_initial_state = latent_dec(latent_vector)

        bucket_outputs, _, beam_path, beam_symbol = decoder(decoder_initial_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        beam_paths.append(beam_path)
        beam_symbols.append(beam_symbol)
        total_size = math_ops.add_n(weights[:bucket[1]])
        total_size += 1e-12
        KL_divergences.append(tf.reduce_mean(kl_cost / total_size))
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses, KL_objs, KL_costs
