import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile
import re

from useful_functions import predict_slots_query, SlotParser
from HODL import TransformerEncoder, PositionalEmbedding

##################### CONSTANTS #####################
keras.utils.set_random_seed(2024)
MAX_QUERY_LENGTH = 50  #size of input

# Read training data
train_df = pd.read_csv("training_data.csv")

train_query = train_df['query'].values
train_slotfilling = train_df['slot filling'].values

transformer_model = keras.models.load_model("sql_transformer.keras", custom_objects={
    "TransformerEncoder": TransformerEncoder,
    "PositionalEmbedding": PositionalEmbedding,
})


stock_data = pd.read_csv('output_df.csv')
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
                                    transformer_model,
                                    vectorize_query_text,
                                    vectorize_slot_text)
    
    return_df, written_query, all_data = SlotParser(slot_filling,
                                                    prompt,
                                                    stock_data)
    return_df.columns = [" ".join([x.capitalize() for x in col.split("_")]) for col in return_df.columns]
    st.data_editor(return_df,
                hide_index=True,
                key="filtered_df",
                use_container_width=True,
                disabled=True)
    
    st.write("Prompt you submitted:")
    st.info(prompt)
    
    st.write("Slot filling procedure (removes number requests):")
    st.info(slot_filling)
    
    st.write("SQL Query:")
    st.info(written_query)
        
    st.write("Entire Table:")
    st.data_editor(all_data,
                hide_index=True,
                use_container_width=True,
                disabled=True)