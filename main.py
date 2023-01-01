import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers.legacy import Adam

def split_label_string(label_string):
    return list(map(int, label_string.split(",")))

def split_input_string(input_string):
    return list([*input_string])
def convert_labels(labels, input_length, num_labels):
    # Initialize an array of zeros with shape (input_length, num_labels)
    label_array = np.zeros((input_length, num_labels))
    # Iterate over the labels
    for i, label in enumerate(labels):
        # Set the value of the corresponding element in the label array to 1
        label_array[i, label] = 1
    return label_array

label_values = {
    "space": 0,
    "city": 1,
    "district": 2,
    "neighborhood": 3,
    "neighborhood_suffix": 4,
    "road": 5,
    "road_suffix": 6,
    "street": 7,
    "street_suffix": 8,
    "building_number_prefix": 9,
    "building_number": 10,
    "floor_suffix": 11,
    "floor": 12,
    "door_number_prefix": 13,
    "door_number": 14,
    "building_block_suffix": 15,
    "building_site_name": 16,
    "building_block": 17,
}

# Read the input data and labels from the CSV files
data = pd.read_csv("address.csv", delimiter='|', header=None, names=["input"], dtype=str)
labels = pd.read_csv("labels.csv", delimiter=']', header=None, names=["label"])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Split the label strings into a list of labels
y_train = y_train["label"].apply(split_label_string)
y_test = y_test["label"].apply(split_label_string)

# Split the input strings into a list of input
X_train = X_train["input"].apply(split_input_string)
X_test = X_test["input"].apply(split_input_string)

# Use a tokenizer to preprocess the input data
tokenizer = Tokenizer(char_level=True, filters=None)
tokenizer.fit_on_texts(X_train)
word_count = len(tokenizer.word_index)
X_train = [tokenizer.texts_to_matrix(y) for y in X_train]
X_test = [tokenizer.texts_to_matrix(y) for y in X_test]

y_train = [convert_labels(y, 128, len(label_values.values())) for y in y_train]
y_test = [convert_labels(y, 128, len(label_values.values())) for y in y_test]

y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Define the LSTM layer
lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))

# Define the model
model = tf.keras.Sequential([
    lstm1,
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(1024, activation="sigmoid"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="sigmoid"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(18, activation="sigmoid")
])

# Compile the model with a categorical cross-entropy loss function and an Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# Fit the model on the input data and labels
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

# Calculate the model's performance on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

model.save('./trained')

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

loaded = tf.keras.models.load_model('./trained')

predictions = loaded.predict(np.array([X_train[1001]]))

print(data['input'][20001])
# Loop through the characters in the input sequence
for i in range(128):
    # Get the predicted label for the current character
    predicted_label = np.argmax(predictions[0][i])

    # Get the label name for the predicted label value
    label_name = [key for key in label_values if label_values[key] == predicted_label][0]

    # Print the character and the label name
    print(f"{[*data['input'][20001]][i]}: {label_name}")