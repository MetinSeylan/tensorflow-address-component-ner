import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
import math
import preprocess


def calculate_spe(y):
    return int(math.ceil((1. * y) / batch_size))


batch_size = 128
chunk_size = 960000
data_len = 8746216
validata_len = 8746216
label_len = 26
input_len = 150

data = preprocess.read_data('dataset.csv', chunk_size=chunk_size)
tokenizer = preprocess.load_tokenizer(data)
vocab_size = len(tokenizer.word_index) + 1

train_gen = preprocess.get_generator(data, tokenizer, chunk_size=chunk_size, batch_size=batch_size, label_len=label_len,
                                     input_size=input_len)

test_gen = preprocess.get_generator(data, tokenizer, chunk_size=chunk_size, batch_size=batch_size, label_len=label_len,
                                    input_size=input_len)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=2048, input_length=150),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(26, activation="softmax")
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

steps_per_epoch = calculate_spe(data_len)
validation_steps = calculate_spe(validata_len)

callback = tf.keras.callbacks.EarlyStopping(patience=3)

model.fit(train_gen, epochs=10, steps_per_epoch=steps_per_epoch, validation_data=test_gen,
          validation_batch_size=batch_size,
          validation_steps=validation_steps,
          callbacks=[callback])

model.evaluate(test_gen)

model.save('./trained')
