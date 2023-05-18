import pandas as pd
import numpy as np

#Loading data
bitcoin_file = "/content/drive/MyDrive/CUMMW/2022_Problem_C_DATA/BCHAIN-MKPRU.csv"
gold_file = "/content/drive/MyDrive/CUMMW/2022_Problem_C_DATA/LBMA-GOLD.csv"
bitcoin_df = pd.read_csv(bitcoin_file)
gold_df = pd.read_csv(gold_file)
bitcoin_df.set_index('Date', inplace=True)
gold_df.set_index('Date', inplace=True)

# merging gold and bitcoin data
result = bitcoin_df.join(gold_df, how='outer')
result.plot()

# adding old index back so "Date" becomes a column once again
result = result.reset_index()
#converting date to datetime format to sort
result['Date'] = pd.to_datetime(result['Date'])
# sorting entries by date
result.sort_values(by='Date', inplace = True)
result = result.reset_index()
result = result.drop('index',axis=1)
#renaming columns
result = result.rename(columns={"Value": "bitcoin-price", "USD (PM)": "gold-price"})
# column t - days since 2016-9-11
result["t"] = (result["Date"]-result["Date"][0]).dt.days
result

#Handing missing gold
print(result.isnull().sum())
# filling gold price with the price AFTER since its easier for now
result = result.fillna(method='bfill', axis=0)
print(result.isnull().sum())

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Bitcoin price over time")
sns.lineplot(data=result[['bitcoin-price']])

# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Gold price over time")
sns.lineplot(data=result[['gold-price']])

# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Gold and bitcoin")
sns.lineplot(data=result[['gold-price', 'bitcoin-price']])

def buy(asset, curr_price, buy_value, assets, cash, comission):
  if buy_value <= cash:
    cash = cash - buy_value
    buy_amount = buy_value/curr_price
    assets[asset] += (1-comission)*buy_amount
    #print(f"bought {buy_amount} of bitcoin at {curr_price} for a total value of {buy_value}")
  else:
    pass
    #print("not enough cash")
  return cash, assets

def sell(asset, curr_price, sell_amount, assets, cash, comission):
  if sell_amount <= assets[asset]:
    sell_value =  sell_amount*curr_price
    cash += (1-comission)*sell_value
    assets[asset] = assets[asset] -  sell_amount
    #print(f"sold {sell_amount} of bitcoin at {curr_price} for a total value of {sell_value}")
  else:
    pass
    #print('not enough bitcoin')
  return cash, assets

def buy_indicator(curr_price, sma, gradient):
  if curr_price < sma and gradient >0:
    indicator = gradient
  else:
    indicator = 0
  return indicator

def sell_indicator(curr_price, sma, comission, margin):
  sell_now_profit = (1-comission)*(curr_price - sma)
  indicator = sell_now_profit - margin
  return indicator

def sigmoid(x):
  return 1/(1 + np.exp(-0.005*x))

def trading_day(t, table, curr_cash, curr_assets, bitcoin_comission, gold_comission):
  bitcoin_comission = 0.01
  gold_comission = 0.02

  # HYPERPARAMETERS
  gold_period = 50 #moving average

  gold_margin = 50
  gold_comission = 0.01

  bitcoin_period = 50
  bitcoin_margin = 100
  bitcoin_comission = 0.02

  gradient_period = 5

  bitcoin_buy_amount_multi = 0.2
  gold_buy_amount_multi = 0.05
  bitcoin_sell_amount_multi = 10
  gold_sell_amount_multi = 0.2

  row = table.loc[t, :]
  # Dont do any purchases until we have enough data
  if t < 6:
    table.loc[t, 'bitcoin-value-at-hand'] = 0
    table.loc[t, 'gold-value-at-hand'] = 0
    table.loc[t, 'cash-at-hand'] = curr_cash
    return curr_cash, curr_assets

  # get current prices
  gold_price = table['gold-price'][t]
  bitcoin_price = table['bitcoin-price'][t]

  # Get gradient over 5 days
  gold_gradient = table['gold-price'][t] - table['gold-price'][t-gradient_period]
  table.loc[t, 'gold-gradient']= gold_gradient

  bitcoin_gradient = table['bitcoin-price'][t] - table['bitcoin-price'][t-gradient_period]
  table.loc[t, 'bitcoin-gradient'] = bitcoin_gradient

  if t>50:
    # getting bitcoin and gold gradient over last 50 days
    gold_50_gradient = table['gold-price'][t] - table['gold-price'][t-50]
    table.loc[t, 'gold-50-gradient']= gold_50_gradient

    bitcoin_50_gradient = table['bitcoin-price'][t] - table['bitcoin-price'][t-50]
    table.loc[t, 'bitcoin-50-gradient'] = bitcoin_50_gradient

    gold_period = int(round(gold_period/gold_50_gradient,0))
    bitcoin_period = int(round(bitcoin_period/bitcoin_50_gradient,0))

  # Dont do any purchases until we have enough data
  if t < gold_period or t < bitcoin_period:
    table.loc[t, 'bitcoin-value-at-hand'] = 0
    table.loc[t, 'gold-value-at-hand'] = 0
    table.loc[t, 'cash-at-hand'] = curr_cash
    return curr_cash, curr_assets

  # calculate moving average over past period days
  gold_sma = table['gold-price'][t-gold_period:t].mean()
  table.loc[t, 'gold-sma']= gold_sma

  bitcoin_sma = table['bitcoin-price'][t-bitcoin_period:t].mean()  
  table.loc[t, 'bitcoin-sma']= bitcoin_sma
  
  # getting buy and sell indicators
  buy_gold_indicator = buy_indicator(gold_price, gold_sma, gold_gradient)
  table.loc[t, 'buy_gold_indicator'] = buy_gold_indicator
  buy_bitcoin_indicator = buy_indicator(bitcoin_price, bitcoin_sma, bitcoin_gradient)
  table.loc[t, 'buy_bitcoin_indicator'] = buy_bitcoin_indicator

  sell_gold_indicator = sell_indicator(gold_price, gold_sma, gold_comission, gold_margin)
  table.loc[t, 'sell_gold_indicator'] = sell_gold_indicator
  sell_bitcoin_indicator = sell_indicator(bitcoin_price, bitcoin_sma, bitcoin_comission, bitcoin_margin)
  table.loc[t, 'sell_bitcoin_indicator'] = sell_bitcoin_indicator

  
  
  # determine what to buy
  if buy_bitcoin_indicator > 0 or buy_gold_indicator > 0:
    if buy_bitcoin_indicator > buy_gold_indicator:
      # to get buy percentage, divide buy bitcoin indicator (same as gradient) by maximum observed gradient and then by 5
      buy_percentage = bitcoin_buy_amount_multi*
      id(buy_bitcoin_indicator)
      #print(buy_percentage)
      buy_value = curr_cash*buy_percentage
      curr_cash, curr_assets = buy(asset='bitcoin',
                                  curr_price=bitcoin_price,
                                  buy_value=buy_value,
                                  cash=curr_cash,
                                  assets=curr_assets,
                                  comission=bitcoin_comission)
    else:
      # to get buy percentage, divide buy bitcoin indicator (same as gradient) by maximum observed gradient and then by 5
      buy_percentage = gold_buy_amount_multi*sigmoid(buy_gold_indicator)
      #print(buy_percentage)
      buy_value = curr_cash*buy_percentage
      curr_cash, curr_assets = buy(asset='gold',
                                  curr_price=gold_price,
                                  buy_value=buy_value,
                                  cash=curr_cash,
                                  assets=curr_assets,
                                  comission=gold_comission)

  if sell_bitcoin_indicator > 0:
    # sell percentage is profit per unit divided by price per unit multiply
    # profit per unit is buying indicator + margin
    sell_percentage = bitcoin_sell_amount_multi*(sell_bitcoin_indicator + bitcoin_margin)/(bitcoin_price)
    sell_amount = curr_assets['bitcoin']*sell_percentage
    curr_cash, curr_assets = sell(asset='bitcoin',
                                  curr_price=bitcoin_price,
                                  sell_amount=sell_amount,
                                  cash=curr_cash,
                                  assets=curr_assets,
                                  comission=bitcoin_comission)
  if sell_gold_indicator > 0:
    # sell percentage is profit per unit divided by price per unit multiply by 5
    # profit per unit is buying indicator + margin
    sell_percentage = gold_sell_amount_multi*(sell_gold_indicator + gold_margin)/gold_price
    sell_amount = curr_assets['gold']*sell_percentage
    curr_cash, curr_assets = sell(asset='gold',
                                  curr_price=gold_price,
                                  sell_amount=sell_amount,
                                  cash=curr_cash,
                                  assets=curr_assets,
                                  comission=gold_comission)
    
  table.loc[t, 'bitcoin-value-at-hand'] = curr_assets['bitcoin']*bitcoin_price
  table.loc[t, 'gold-value-at-hand'] = curr_assets['gold']*gold_price
  table.loc[t, 'cash-at-hand'] = curr_cash

  return curr_cash, curr_assets

def simulate(table):
  bitcoin_comission = 0.02
  gold_comission = 0.01
  
  cash = 10000
  assets = {'bitcoin':0, 'gold':0}

  for t in range(len(table)):
    cash,assets = trading_day(t, table, cash, assets, bitcoin_comission, gold_comission)
    if t % 100 == 0:
      #print(f"bitcoin_amount:{assets['bitcoin']}")
      #print(f"bitcoin_price:{table['bitcoin-price'][t]}")
      print(f"day {t} cash: {table['cash-at-hand'][t]}, gold-value: {table['gold-value-at-hand'][t]}, bitcoin-value: {table['bitcoin-value-at-hand'][t]}")
      
       
      
  #sell anything remaining in the end - not needed anymore
  """
  cash, assets = sell(asset='bitcoin', curr_price=table.iloc[-1]['bitcoin-price'],
                                       sell_amount=assets['bitcoin'],
                                       cash=cash,
                                       assets=assets,
                                       comission=bitcoin_comission)
  
  cash, assets = sell(asset='gold', curr_price=table.iloc[-1]['gold-price'],
                                       sell_amount=assets['gold'],
                                       cash=cash,
                                       assets=assets,
                                       comission=bitcoin_comission)
  """

  final_value = table['cash-at-hand'][len(table)-1]+table['gold-value-at-hand'][len(table)-1]+table['bitcoin-value-at-hand'][len(table)-1]
  print(f"Final value at hand: {round(final_value,2)} which gives {round(final_value/100,2)} % ROI")
  
  simulate(result)
  
  stackplot_data = result[['cash-at-hand', 'bitcoin-value-at-hand', 'gold-value-at-hand']]

  stackplot_data.plot.area(figsize=(12, 6), colormap='Paired')
plt.figure(figsize=(12,6))
plt.style.use('seaborn-darkgrid')
sns.lineplot(data=result[['bitcoin-price']])

# bitcoin price and buy days chart

#buy_bitcoin_days = list(result.loc[result['buy_bitcoin_indicator']>0]['t'])
#for day in buy_bitcoin_days:
  #plt.axvline(day,color='g')
sns.lineplot(data=result[['bitcoin-price']])

result['buy_amount']= 0.2*result[['buy_bitcoin_indicator']].apply(sigmoid)

# bitcoin price and buy days chart
plt.figure(figsize=(12,6))
plt.style.use('seaborn-darkgrid')
#buy_bitcoin_days = list(result.loc[result['buy_bitcoin_indicator']>0]['t'])
#for day in buy_bitcoin_days:
  #plt.axvline(day,color='g')
sns.lineplot(data=result[['buy_bitcoin_indicator']])
plt.figure(figsize=(12,6))
plt.style.use('seaborn-darkgrid')
sns.lineplot(data=result[['buy_amount']])

plt.figure(figsize=(20,10))
#buy_bitcoin_days = list(result.loc[result['buy_bitcoin_indicator']>0]['t'])
#for day in buy_bitcoin_days:
  #plt.axvline(day,color='g')
sns.lineplot(data=result[['buy_bitcoin_indicator', 'sell_bitcoin_indicator']])

plt.figure(figsize=(20,10))
#buy_bitcoin_days = list(result.loc[result['buy_bitcoin_indicator']>0]['t'])
#for day in buy_bitcoin_days:
  #plt.axvline(day,color='g')
sns.lineplot(data=result[['buy_gold_indicator', 'sell_gold_indicator']])

plt.figure(figsize=(20,10))
#buy_bitcoin_days = list(result.loc[result['buy_bitcoin_indicator']>0]['t'])
#for day in buy_bitcoin_days:
  #plt.axvline(day,color='g')
sns.lineplot(data=result[['buy_bitcoin_indicator', 'buy_gold_indicator']])
