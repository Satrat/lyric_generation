import pandas as pd
import os
import numpy as np
import tensorflow.compat.v1 as tf

LYRICS_DATA_DIR = './data'
LYRICS_CHECKPOINT_DIR = './training_checkpoints'
LATEST_CHECKPOINT_DIR = './best_checkpoint'
LYRICS_FILE = 'lyrics.txt'

vocab_size = 38
embedding_dim = 256
rnn_units = 512

def get_vocab_maps(text):
    vocab = sorted(set(text))
    global vocab_size
    vocab_size = len(vocab)
    
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

def generate_text(model, start_string,t, length_char):
    # Evaluation step (generating text using the learned model)
    text = open('lyrics.txt', 'rb').read().decode(encoding='utf-8')
    char2idx, idx2char = get_vocab_maps(text)

    # Number of characters to generate
    num_generate = length_char

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

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
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    result = (start_string + ''.join(text_generated))
    result = result.replace('  ', ' ')
    return result.lower()[:-1] #skip last word since its probably incomplete

def build_inf_model():
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(LATEST_CHECKPOINT_DIR))
    model.build(tf.TensorShape([1, None]))
    return model

def generate_text(model, lyrics_path, start_string,t, length_char):
    # Evaluation step (generating text using the learned model)
    text = open(lyrics_path, 'rb').read().decode(encoding='utf-8')
    char2idx, idx2char = get_vocab_maps(text)

    # Number of characters to generate
    num_generate = length_char

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

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
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    result = (start_string + ''.join(text_generated))
    result = result.replace('  ', ' ')
    return result

def build_inf_model(path_to_checkpoint):
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(path_to_checkpoint))
    model.build(tf.TensorShape([1, None]))
    return model