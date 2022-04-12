import numpy as np
import tensorflow as tf

LYRICS_DATA_DIR = './data'
LYRICS_CHECKPOINT_DIR = './training_checkpoints_250_batch'
LATEST_CHECKPOINT_DIR = './best_checkpoint'
LYRICS_FILE = 'lyrics.txt'

vocab_size = 38
embedding_dim = 256
rnn_units = 512

def get_vocab_maps(text):
    vocab = sorted(set(text))
    #print('{} unique characters'.format(len(vocab)))
    assert(len(vocab) == vocab_size)
    
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    return char2idx, idx2char

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model