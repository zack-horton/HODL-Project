import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, MultiHeadAttention, LayerNormalization
# from tensorflow.keras import Model, Sequential

test_df = pd.read_csv("testing_data.csv")
train_df = pd.read_csv("training_data.csv")

train_query = train_df['query'].values
train_slotfilling = train_df['slot filling'].values
test_query = test_df['query'].values
test_slotfilling = test_df['slot filling'].values

# keras.utils.set_random_seed(2024)