import pandas as pd
import os
import numpy as np
import tensorflow.compat.v1 as tf
from shared import *

BATCH_SIZE = 250
BUFFER_SIZE = 10000
EPOCHS = 150
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

def cleaning(df):
    a=[]
    i=0
    df1=df
    title = df['Title']
    for t in df['Title']:
        r=Re=l=Li=c=m=V=ve=D=rs=0
        r=t.find('remix')
        Re=t.find('Remix')
        l=t.find('live')
        Li=t.find('Live')
        V=t.find('Version')
        ve=t.find('version')
        D=t.find('Demo ')
        D=t.find('Demo')
        rs=t.find('Reprise')
        c=t.find('COPY')
        m=t.find('Mix')
        if r != -1:
            a.append(t)
        elif Re != -1:
            a.append(t)
        elif l != -1:
            a.append(t)
        elif Li != -1:
            a.append(t)
        elif V != -1:
            a.append(t)
        elif ve != -1:
            a.append(t)
        elif D != -1:
            a.append(t)
        elif rs != -1:
            a.append(t)
        elif c != -1:
            a.append(t)
        elif m != -1:
            a.append(t)
    
    for t1 in df['Title']:
        for t2 in a:
            if t1 == t2:
                df1=df1.drop(i)
        i=i+1
    
    df1.dropna(subset = ["Title"], inplace=True)
    df1.dropna(subset = ["Lyric"], inplace=True)
    df1.drop_duplicates(subset ="Title",keep = False, inplace = True)
    df1.drop_duplicates(subset ="Lyric",keep = False, inplace = True)    
    
    return df1

def preprocess_lyrics(lyrics_folder):
    dfs = []
    for file in os.listdir(lyrics_folder):
        if file.endswith(".csv"):
            fp = os.path.join(lyrics_folder, file)
            df = pd.read_csv(fp)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            df = df[['Artist', 'Title', 'Album', 'Year', 'Date', 'Lyric']]
            dfs.append(cleaning(df))
    all_lyrics = pd.concat(dfs, axis=0, ignore_index=True)
    
    final_str = ""
    removed = 0
    for i in range(all_lyrics.Lyric.size):
        if not all_lyrics.Lyric[i].isascii():
            removed += 1
        else:
            single_lyric = all_lyrics.Lyric[i].replace('   ',' \n ') + ' \n '
            final_str += single_lyric
    print("Removed {} non-ascii lyrics out of {}".format(removed, all_lyrics.Lyric.size))
    return final_str

def get_text_as_int(text, char2idx):
    text_as_int = np.array([char2idx[c] for c in text])
    return text_as_int

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

if __name__ == "__main__":
    text = preprocess_lyrics(LYRICS_DATA_DIR)
    f = open(LYRICS_FILE, "w")
    f.write(text)
    f.close()
    
    char2idx, idx2char = get_vocab_maps(text)
    text_as_int = get_text_as_int(text, char2idx)
    
    # The maximum length sentence in characters
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    #build model
    model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    model.summary()

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    model.compile(optimizer='adam', loss=loss)
    
    # Directory where the checkpoints will be saved
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(LYRICS_CHECKPOINT_DIR, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])