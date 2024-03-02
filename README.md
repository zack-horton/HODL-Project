# HODL-Project

Project developed as part of course: 15.773: Hands-on Deep Learning Project (Spring 2024)

Collaborators: [Zack Horton](https://github.com/zack-horton), [Tanner Street](), Virginia Maguire, and Yutong Zhang

**Description:**

The primary goal of this project was to explore new and exciting applications of Deep Learning, expanding on the resources and  information learned during class. The application we chose was to analyze stock market prices for the S&P 500 through LSTMs, create an aggregate summary table (ideally utilized by a day-trader) from the LSTM forecasts, which could be queried through our transformer-based model via our streamlit application (see `webapp.py`). The transformer model was trained to conduct slot-filling, similar to the one that we discussed during class used for the ATIS dataset (more on that project described [here](https://catalog.ldc.upenn.edu/docs/LDC93S4B/corpus.html)). The training dataset (see `training_data.csv` and  `testing_data.csv`) was created through hand-labeling the questions related to our dataset. The training questions were generated via ChatGPT 3.5 after prompting it to create permutations of specific questions, those prompt questions were used as the test set data).


We trained a keras based-model, using some sample code for the Transformer Encoder and Position Embedding layer provided to us by our course administrators (see `HODL.py`). This was very similar to the implementation in the famous paper, [Attention Is All You Need](https://doi.org/10.48550/arXiv.1706.03762). This model is saved in two locations, `sql_transformer.keras` and `model_save.zip`. However, in order to successfully interact and query the dataset we are compiling, we created the `slot_filling.py` file, and implemented the `SlotParser` in order to translate the tokens outputted from the transformer model into a usable SQL query (and then execute that on the pandas dataframe in which the aggregate table is saved).


**Model Performance:**

After model training was complete, we evaluated it on the test-set performance. For the specific slots (not the individual tokens, as we exclude the empty/null `o` tokens) we achieved around a 93% accuracy. To demonstrate the most accurate capabilities of our model, we created the `demo_questions.txt` file, to provide some prompts that achieve perfect execution. It is an excellent outcome considering our relatively small amount of training data (2,364 observations). However, the initial labeling was only 394 rows, and the rest of the data was created through a find-and-replace process, substituting synonymous words with one another (forecast $\rightarrow$ prediction, greatest $\rightarrow$ largest, smallest $\rightarrow$ lowest, etc.). This was done in an effort to enrich the vocabulary of our training corpus.
