import io
import json
from os import path
import pandas as pd
import tensorflow as tf
import numpy as np


def parse_label_data(label_string):
    return list(map(int, label_string.split(",")))


def parse_input_data(input_string):
    return list([*input_string])


def convert_labels(labels, input_length, num_labels):
    # Initialize an array of zeros with shape (input_length, num_labels)
    label_array = np.zeros((input_length, num_labels))
    # Iterate over the labels
    for i, label in enumerate(labels):
        # Set the value of the corresponding element in the label array to 1
        label_array[i, label] = 1
    return label_array


def read_data(filename, chunk_size=32000):
    return pd.read_csv(filename, header=None, names=["input", "label"], dtype=str, iterator=True, chunksize=chunk_size)


def load_tokenizer(dataset):
    if dataset and path.exists("tokenizer.json") is False:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters=None)
        tokenizer.fit_on_texts(dataset)
        tokenizer_json = tokenizer.to_json()
        with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        return tokenizer
    else:
        with open('tokenizer.json') as f:
            data = json.load(f)
            return tf.keras.preprocessing.text.tokenizer_from_json(data)


def get_generator(data, tokenizer, chunk_size=32000, batch_size=32, label_len=26, input_size=150):
    while True:
        chunk = data.get_chunk()
        if chunk.size == 0:
            break

        for offset in range(0, chunk_size, batch_size):
            print('\n', offset)
            y_train = chunk["label"].iloc[offset: offset + batch_size]
            x_train = chunk["input"].iloc[offset: offset + batch_size]

            y_train = y_train.apply(parse_label_data)
            x_train = x_train.apply(parse_input_data)

            batch_data = tokenizer.texts_to_sequences(x_train)
            batch_labels = [convert_labels(y, input_size, label_len) for y in y_train]

            yield np.array(batch_data), np.array(batch_labels)



