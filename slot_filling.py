import pandas as pd
from useful_functions import SlotParser

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

for (slot, prompt) in zip(example_slots, example_prompts):
    print(slot)
    print(prompt)
    SlotParser(slot, prompt, data)
    print()