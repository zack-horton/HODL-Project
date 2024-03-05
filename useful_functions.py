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
    slot_filling = slot_filling.strip()  #strip any whitespace from slot_filling return
    # initialize aspects of the SQL query
    slots = {'select': [],
             'order': [],
             'limit': None}
    
    # for word (token) in prompt
    for word in prompt.split(" "):
        if word.isnumeric():  #if we found a number
            slots['limit'] = int(word)  #assume number relates to LIMIT (# of rows to display)
    
    # for slot (token) in output
    for token in slot_filling.split(" "):
        if (token != "o") and (token != "O"):  #if not a null slot
            if 'select-' in token:  #if a select slot
                if token not in slots['select']:  #if we haven't already added it
                    slots['select'].append(token)  #then include in SELECT statement
            elif 'order-by' in token:  #if an order-by slot
                if token not in slots['order']:  #if we haven't already added it
                    slots['order'].append(token)  #then include in ORDER BY statement

    if len(slots['select']) == 0:  #if we haven't selected a colummn
        columns = list(stock_data.columns)  #assume all
    else:  #if we have selected >= 1 column
        columns = [x.split("-")[1] for x in slots['select']]  #access column names, format them
    
    if len(slots['order']) == 0:  #if we haven't ordered a column
        if 'stock' not in columns:
            columns.insert(0, 'stock')  #insert at beginning
        order_cols = ['stock']  #assume we order by stock
        order_ascending = [True]  #assum ascending order
    else:  #if there is order statement
        order_cols = [x.split("-")[2] for x in slots['order']]  #format
        order_ascending = [True if x.split("-")[-1] == 'asc' else False for x in slots['order']]  #format

    if slots['limit'] == None:  #if we don't have a number
        limit = len(stock_data)  #assume we want all rows
    else:  #if there is a limit number
        limit = int(slots['limit'])  #make sure to return that many rows
        
    for col in order_cols:  #for every ordering column
        if col not in columns:  #check if it is an accessible column
            columns.append(col)  #if not, add it
    
    pandas_query = stock_data[columns]  #use only selected columns
    pandas_query = pandas_query.sort_values(by=order_cols,  #columns to sort
                                            ascending=order_ascending,  #boolean list of asc/desc
                                            ignore_index=True)
    pandas_query = pandas_query.head(limit)  #LIMIT statement
    
    all_data = stock_data.sort_values(by=order_cols,
                                      ascending=order_ascending,
                                      ignore_index=True)  #include just in case
    # print(pandas_query)  #print the dataframe

    #formatting SQL statements
    SELECT = f"SELECT {', '.join(columns)}"
    FROM = "FROM stock_data"
    ORDER_BY = f"ORDER BY {', '.join((str(x.split('-')[2])+' '+str(x.split('-')[-1].upper())) for x in slots['order'])}"
    LIMIT = f"LIMIT {slots['limit']}"

    #checking for errors
    if len(slots['select']) == 0:  #empty slots
        SQL_QUERY = "SELECT *\n FROM stock_data"
    elif (len(slots['order']) == 0) and (slots['limit'] == None):  #no sort or limit statement
        SQL_QUERY = f"{SELECT}\n{FROM}"
    elif len(slots['order']) == 0:  #no order clause
        SQL_QUERY = f"{SELECT}\n{FROM}\n{LIMIT}"
    elif slots['limit'] == None:  #no limit clause
        SQL_QUERY = f"{SELECT}\n{FROM}\n{ORDER_BY}"
    else:
        SQL_QUERY = f"{SELECT}\n{FROM}\n{ORDER_BY}\n{LIMIT}"
    
    print(SQL_QUERY)  #print SQL statement
    
    return pandas_query, SQL_QUERY, all_data