import pandas as pd

example_slots = [
    "o select-stock o o o o o order-by-forecasted_price_change-desc select-forecasted_price o o o o"
    "o select-stock o select-forecasted_volatility o select-forecasted_volatility o o o order-by-forecasted_volatility-desc select-forecasted_volatility o o o",
    "o select-stock o o o o o o o order-by-percent_change-asc select-percent_change select-percent_change o o",
    "o select-stock o o o o o order-by-forecasted_price-desc select-forecasted_price o order-by-forecasted_volatility-asc select-forecasted_volatility"
]
example_prompts = [
    "What 5 stocks are expected to have the highest increase in price for tomorrow?",
    "What 10 stocks are forecasted or predicted to have the highest volatility in their price?",
    "What 2 stocks are forecasted or predicted to have the lowest percent change in price?",
    "What 8 stocks are predicted to have the highest price and lowest volatility"
]
data = pd.read_csv("prototype_table.csv")

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

    SELECT = f"SELECT {', '.join([x.split('-')[1] for x in slots['select']])}"
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
    print(pandas_query)

    return pandas_query, SQL_QUERY

for (slot, prompt) in zip(example_slots, example_prompts):
    print(slot)
    print(prompt)
    SlotParser(slot, prompt, data)
    print()