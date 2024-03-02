import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile
import re

from NLP_to_SQL import predict_slots_query
from slot_filling import SlotParser

##################### CONSTANTS #####################
keras.utils.set_random_seed(2024)
MAX_QUERY_LENGTH = 50  #size of input

# Read training data
train_df = pd.read_csv("training_data.csv")

train_query = train_df['query'].values
train_slotfilling = train_df['slot filling'].values

with ZipFile('model_save.zip', 'r') as zip:
    zip.extractall()
MODEL = keras.models.load('nlp_to_sql.keras')

stock_data = pd.read_csv('prototype_table.csv')
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

vectorize_slot_text.adapt(train_slotfilling)  #build slot vocabulary
slots_train = vectorize_slot_text(train_slotfilling)  #vectorized training slots

#### BEGIN STREAMLIT APPLICATION ####
st.title("Day-Trader GPT")
st.write()
st.write("Enter your question into the following textbox:")

prompt = st.text_area(label="Prompt:",
                      value="")
run_query_button = st.button("Run query",
                             type="primary")

if run_query_button:
    formatted_prompt = re.sub(r'[0-9]', '', prompt)
    slot_filling = predict_slots_query(formatted_prompt,
                                       MODEL,
                                       vectorize_query_text,
                                       vectorize_slot_text)
    
    return_df, written_query = SlotParser(slot_filling,
                                                prompt,
                                                stock_data)
    return_df.columns = [" ".join([x.capitalize() for x in col.split("_")]) for col in return_df.columns]
    st.data_editor(return_df,
                   hide_index=True,
                   use_container_width=True,
                   disabled=True)
    
    st.info(f"Your prompt:\n{prompt}")
    st.info(f"Tokenization, via slot filling:\n{slot_filling}")
    st.info(f"Query we calculated for you:\n{written_query}")
    
    st.data_editor(stock_data,
                   hide_index=True,
                   use_container_width=True,
                   disabled=True)