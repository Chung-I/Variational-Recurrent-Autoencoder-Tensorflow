import numpy as np
import h5py
import re
from tensorflow.python.platform import gfile
import pdb
import sys



def get_vocab_order_embeddings(vocab_path, word2vec_path):
  def load_word2vec(word2vec_path):
    def str2embed(embed_strs):
      return [float(num) for num in embed_strs]

    with gfile.GFile(word2vec_path, "r") as w2v_file:
      raw_embeddings = [re.split("\s+", line)[:-1] for line in w2v_file.readlines()]
      embedding_dict = {embed[0]: str2embed(embed[1:]) for embed in raw_embeddings[1:]}
    return embedding_dict

  embed_dict = load_word2vec(word2vec_path)
  with gfile.GFile(vocab_path, "r") as vocab_file:
    vocab_list = [line.strip("\n") for line in vocab_file.readlines()]
    embed_list = [embed_dict.get(word, [0]*len(embed_dict["\u3002"])) for word in vocab_list]
  return embed_list

embed_list = get_vocab_order_embeddings(sys.argv[1], sys.argv[2])
embeddings = np.array(embed_list, dtype=np.float32)
with h5py.File(sys.argv[1] + '.embeddings.h5', 'w') as h5f:
  h5f.create_dataset('embeddings', data=embeddings)
