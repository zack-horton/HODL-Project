import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile

import tensorflow as tf
from tensorflow import keras

from HODL import TransformerEncoder, PositionalEmbedding

##################### CONSTANTS #####################
keras.utils.set_random_seed(2024)
MAX_QUERY_LENGTH = 50  #size of input
EMBED_DIM = 512  #dimension of embeddings
DENSE_DIM = 128
NUM_HEADS = 8  #number of multi-attention heads
DENSE_UNITS = 128  #num nodes in hidden layer
BATCH_SIZE = 64  #batch size for training transformer
EPOCHS = 10  #epochs for training transformer

# Read training data
test_df = pd.read_csv("testing_data.csv")
train_df = pd.read_csv("training_data.csv")

train_query = train_df['query'].values
train_slotfilling = train_df['slot filling'].values
test_query = test_df['query'].values
test_slotfilling = test_df['slot filling'].values
#####################################################

# CREATE VECTORIZER (QUERY & SLOTS)
vectorize_query_text = keras.layers.TextVectorization(
    max_tokens=None,  #no maximum vocabulary
    output_sequence_length=MAX_QUERY_LENGTH,  #pad or truncate output to value
    output_mode="int",  #vector has index of vocabulary
    standardize="lower_and_strip_punctuation",  #convert input to lowercase and rmv punctuation
    split="whitespace",  #split values based on whitespace
    ngrams=1  #only look at whole words
)
vectorize_slot_text = keras.layers.TextVectorization(
    max_tokens=None,  #no maximum vocabulary
    output_sequence_length=MAX_QUERY_LENGTH,
    output_mode="int",  #vector has index of vocabulary
    standardize="lower",  #convert input to lowercase [can't do punctuation b/c of dashes]
    split="whitespace",  #split values based on whitespace
    ngrams=1  #only look at whole words
)

# CREATE VOCABULARY AND VECTORIZED TRAINING DATA
vectorize_query_text.adapt(train_query)  #build vocabulary
query_train = vectorize_query_text(train_query)  #vectorized training queries
query_test = vectorize_query_text(test_query)  #vectorized testing queries
QUERY_VOCAB_SIZE = vectorize_query_text.vocabulary_size() #total vocabulary of queries

vectorize_slot_text.adapt(train_slotfilling)  #build slot vocabulary
slots_train = vectorize_slot_text(train_slotfilling)  #vectorized training slots
slots_test = vectorize_slot_text(test_slotfilling)  #vectorized testing slots
SLOT_VOCAB_SIZE = vectorize_slot_text.vocabulary_size()  #total vocabulary of slots

# BUILD KERAS MODEL
inputs = keras.Input(shape=(MAX_QUERY_LENGTH,))
embedding = PositionalEmbedding(MAX_QUERY_LENGTH,
                                QUERY_VOCAB_SIZE,
                                EMBED_DIM)
x = embedding(inputs)
encoder_out = TransformerEncoder(EMBED_DIM,
                                 DENSE_DIM,
                                 NUM_HEADS)(x)
x = keras.layers.Dense(DENSE_UNITS, activation="relu", name="Dense_Layer")(encoder_out)
x = keras.layers.Dropout(0.25, name="Dropout_Layer")(x)
outputs = keras.layers.Dense(SLOT_VOCAB_SIZE, activation="softmax", name="Softmax_Layer")(x)

model = keras.Model(inputs, outputs)
print()
print(model.summary())
print()

# TRAIN KERAS MODEL
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])
history = model.fit(query_train, slots_train,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS)

# OUT-OF-SAMPLE TESTING
model.evaluate(query_test, slots_test)

# SAVE MODEL
filename = 'nlp_to_sql.keras'
# model.save(filename)
# ZipFile('model_save.zip', mode='w').write(filename)

# EVALUATING SLOT ACCURACY
def slot_filling_accuracy(actual, predicted, only_slots=False):
    not_padding = np.not_equal(actual, 0) #+ np.not_equal(predicted, 0)

    if only_slots:
        non_slot_token = vectorize_slot_text(['O']).numpy()[0, 0]
        slots = np.not_equal(actual, non_slot_token)
        correct_predictions = np.equal(actual, predicted)[not_padding * slots]
    else:
        correct_predictions = np.equal(actual, predicted)[not_padding]

    sample_length = len(correct_predictions)

    weights = np.ones(sample_length)

    return np.dot(correct_predictions, weights) / sample_length

predicted = np.argmax(model.predict(query_test), axis=-1).reshape(-1)
actual = slots_test.numpy().reshape(-1)

acc = slot_filling_accuracy(actual, predicted, only_slots=False)
acc_slots = slot_filling_accuracy(actual, predicted, only_slots=True)

print(f'Accuracy = {acc:.3f}')
print(f'Accuracy on slots = {acc_slots:.3f}')

# TEST-SET EVALUATION
def predict_slots_query(query, model, query_vectorizer, slot_vectorizer):
    sentence = query_vectorizer([query])

    prediction = np.argmax(model.predict(sentence), axis=-1)[0]

    inverse_vocab = dict(enumerate(slot_vectorizer.get_vocabulary()))
    decoded_prediction = " ".join(inverse_vocab[int(i)] for i in prediction)
    return decoded_prediction

for example, answer in zip(test_query, test_slotfilling):
    print()
    print("Query:\n", example)
    print("Answer:\n", answer)
    print("Prediction:\n", predict_slots_query(example, 
                                               model, 
                                               vectorize_query_text, 
                                               vectorize_slot_text))
    print()