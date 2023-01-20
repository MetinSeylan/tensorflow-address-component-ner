import tensorflow as tf
import numpy as np
import json
import preprocess

tokenizer = preprocess.load_tokenizer(None)

with open('labels.json') as f:
    label_values = json.load(f)

print('model loading')
loaded = tf.keras.models.load_model('./trained')
print('model loaded')


def predict(address):
    padded = address.ljust(150).lower()
    print(address)

    predictions = loaded.predict([tokenizer.texts_to_sequences([*padded])])

    for i in range(len(address)):
        predicted_label = np.argmax(predictions[0][i])

        # Get the label name for the predicted label value
        label_name = [key for key in label_values if label_values[key] == predicted_label][0]

        # Print the character and the label name
        print(f"{[*address][i]}: {label_name}")


predict('atasehir')
