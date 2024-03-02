import pandas as pd
import numpy as np
import tensorflow as tf

def predict_slots_query(query, model, query_vectorizer, slot_vectorizer):
    sentence = query_vectorizer([query])

    prediction = np.argmax(model.predict(sentence), axis=-1)[0]

    inverse_vocab = dict(enumerate(slot_vectorizer.get_vocabulary()))
    decoded_prediction = " ".join(inverse_vocab[int(i)] for i in prediction)
    return decoded_prediction

def SlotParser(slot_filling, prompt, stock_data):
    slot_filling = slot_filling.strip()
    slots = {'select': [],
             'order': [],
             'limit': None}
    
    for word in prompt.split(" "):
        if (word != "o") and (word != "O"):
            if word.isnumeric():
                slots['limit'] = int(word)
    
    for token in slot_filling.split(" "):
        if (token != "o") and (token != "O"):
            if 'select-' in token:
                if token not in slots['select']:
                    slots['select'].append(token)
            elif 'order-by' in token:
                if token not in slots['order']:
                    slots['order'].append(token)

    if len(slots['select']) == 0:
        columns = list(stock_data.columns)
    else:
        columns = [x.split("-")[1] for x in slots['select']]
    
    if len(slots['order']) == 0:
        order_cols = ['stock']
        order_ascending = [True]
    else:
        order_cols = [x.split("-")[2] for x in slots['order']]
        order_ascending = [True if x.split("-")[-1] == 'asc' else False for x in slots['order']]

    if slots['limit'] == None:
        limit = len(stock_data)
    else:
        limit = int(slots['limit'])
        
    for col in order_cols:
        if col not in columns:
            columns.append(col)
    
    pandas_query = stock_data[columns]
    pandas_query = pandas_query.sort_values(by=order_cols,
                                            ascending=order_ascending,
                                            ignore_index=True)
    pandas_query = pandas_query.head(limit)
    
    all_data = stock_data.sort_values(by=order_cols,
                                      ascending=order_ascending,
                                      ignore_index=True)
    print(pandas_query)


    SELECT = f"SELECT {', '.join(columns)}"
    FROM = "FROM stock_data"
    ORDER_BY = f"ORDER BY {', '.join((str(x.split('-')[2])+' '+str(x.split('-')[-1].upper())) for x in slots['order'])}"
    LIMIT = f"LIMIT {slots['limit']}"

    if len(slots['select']) == 0:  #empty slots
        SQL_QUERY = "SELECT *\n FROM stock_data"
    elif (len(slots['order']) == 0) and (slots['limit'] == None):  #no sort or limit statement
        SQL_QUERY = f"{SELECT}\n{FROM}"
    elif len(slots['order']) == 0:
        SQL_QUERY = f"{SELECT}\n{FROM}\n{LIMIT}"
    elif slots['limit'] == None:
        SQL_QUERY = f"{SELECT}\n{FROM}\n{ORDER_BY}"
    else:
        SQL_QUERY = f"{SELECT}\n{FROM}\n{ORDER_BY}\n{LIMIT}"
    
    print(SQL_QUERY)
    
    return pandas_query, SQL_QUERY, all_data