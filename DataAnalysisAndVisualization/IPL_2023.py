import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# replace "your_api_key" with your generated key
OPENAI_API_KEY = "your_api_key"
llm = OpenAI(api_token=OPENAI_API_KEY)
pandas_ai = PandasAI(llm)

df = pd.read_csv('IPL_Squad_2023_Auction_Dataset.csv')
print(df.shape)
df.head()

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.head()

pandas_ai.run(df, prompt="Which players are the most costliest buys?")

prompts = """
Which players were the cheapest buys this season and which team bought them?
"""
pandas_ai.run(df, prompt=prompts)

prompts = """
Draw a Bargraph showing How much money was spent by each team this season overall.
"""
pandas_ai.run(df, prompt=prompts)

pandas_ai.run(df, prompt="How many bowler remained unsold and what was their base price?")

pandas_ai.run(df, prompt="How many players remained unsold this season?")

pandas_ai.run(df, prompt="Which type of players were majorly unsold?")

pandas_ai.run(df, prompt="Who are three new players Gujrat picked?")

pandas_ai.run(df, prompt="What is total money spent by all teams in dollars?")

prompts = """
draw a barplot showing 
how much money was spent by Mumbai Indians on all types of players?
"""
pandas_ai.run(df, prompt=prompts)

prompts = """
draw a barplot showing how much money was spent by Gujrat on all types of players?
"""
pandas_ai.run(df, prompt=prompts)

pandas_ai.run(df, prompt="Can you predict which team will buy Sam Curran in 2024?")

print(pandas_ai.run(df, prompt="Can you predict which team will spend the highest money in 2024?"))

pandas_ai.run(df, prompt="Perform univariate analysis")

pandas_ai.run(df, prompt="Perform multivariate analysis")