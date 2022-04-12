import pandas as pd
import os
import numpy as np
import tensorflow as tf2

LYRICS_DATA_DIR = './data'
LYRICS_CHECKPOINT_DIR = './training_checkpoints'
LATEST_CHECKPOINT_DIR = './best_checkpoint'
LYRICS_FILE = 'lyrics.txt'

vocab_size = 37
embedding_dim = 256
rnn_units = 512

def get_vocab_maps(text):
    vocab = sorted(set(text))
    assert(vocab_size == len(vocab))
    
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    return char2idx, idx2char

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf2.keras.Sequential([
        tf2.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf2.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf2.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf2.keras.layers.Dense(vocab_size)
    ])
    return model

def build_inf_model(path_to_checkpoint):
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf2.train.latest_checkpoint(path_to_checkpoint)).expect_partial()
    model.build(tf2.TensorShape([1, None]))
    return model

def generate_text(model, lyrics_path, start_string,t, length_char):
    # Evaluation step (generating text using the learned model)
    text = open(lyrics_path, 'rb').read().decode(encoding='utf-8')
    text = text.replace('\r\n', '\n')
    char2idx, idx2char = get_vocab_maps(text)

    # Number of characters to generate
    num_generate = length_char

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf2.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = t

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf2.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf2.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf2.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    result = (start_string + ''.join(text_generated))
    result = result.replace('  ', ' ')
    return result