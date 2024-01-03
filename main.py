import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# bitcoin_file = 'uploads/BCHAIN-MKPRU.csv'
# gold_file = 'uploads/LBMA-GOLD.csv'

def load_data(bitcoin_file, gold_file):
    bitcoin_df = pd.read_csv(bitcoin_file)
    gold_df = pd.read_csv(gold_file)
    bitcoin_df.set_index('Date', inplace=True)
    gold_df.set_index('Date', inplace=True)
    result = bitcoin_df.join(gold_df, how='outer')
    result = result.reset_index()
    result['Date'] = pd.to_datetime(result['Date'])
    result.sort_values(by='Date', inplace=True)
    result = result.reset_index()
    result = result.drop('index', axis=1)
    result = result.rename(columns={"Value": "bitcoin-price", "USD (PM)": "gold-price"})
    result["t"] = (result["Date"] - result["Date"][0]).dt.days
    result = result.fillna(method='bfill', axis=0)
    return result


def buy(asset, curr_price, buy_value, assets, cash, comission):
    if buy_value <= cash:
        cash = cash - buy_value
        buy_amount = buy_value / curr_price
        assets[asset] += (1 - comission) * buy_amount
        # print(f"bought {buy_amount} of bitcoin at {curr_price} for a total value of {buy_value}")
    else:
        pass
        # print("not enough cash")
    return cash, assets


def sell(asset, curr_price, sell_amount, assets, cash, comission):
    if sell_amount <= assets[asset]:
        sell_value = sell_amount * curr_price
        cash += (1 - comission) * sell_value
        assets[asset] = assets[asset] - sell_amount
        # print(f"sold {sell_amount} of bitcoin at {curr_price} for a total value of {sell_value}")
    else:
        pass
        # print('not enough bitcoin')
    return cash, assets

def sigmoid(x):
  return 1/(1 + np.exp(-0.005*x))


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


def trading_day(t, table, curr_cash, curr_assets, bitcoin_comission, gold_comission):
    bitcoin_comission = 0.01
    gold_comission = 0.02

    # HYPERPARAMETERS
    gold_period = 50  # moving average

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
    gold_gradient = table['gold-price'][t] - table['gold-price'][t - gradient_period]
    table.loc[t, 'gold-gradient'] = gold_gradient

    bitcoin_gradient = table['bitcoin-price'][t] - table['bitcoin-price'][t - gradient_period]
    table.loc[t, 'bitcoin-gradient'] = bitcoin_gradient

    if t > 50:
        # getting bitcoin and gold gradient over last 50 days
        gold_50_gradient = table['gold-price'][t] - table['gold-price'][t - 50]
        table.loc[t, 'gold-50-gradient'] = gold_50_gradient

        bitcoin_50_gradient = table['bitcoin-price'][t] - table['bitcoin-price'][t - 50]
        table.loc[t, 'bitcoin-50-gradient'] = bitcoin_50_gradient

        gold_period = int(round(gold_period / gold_50_gradient, 0))
        bitcoin_period = int(round(bitcoin_period / bitcoin_50_gradient, 0))

    # Dont do any purchases until we have enough data
    if t < gold_period or t < bitcoin_period:
        table.loc[t, 'bitcoin-value-at-hand'] = 0
        table.loc[t, 'gold-value-at-hand'] = 0
        table.loc[t, 'cash-at-hand'] = curr_cash
        return curr_cash, curr_assets

    # calculate moving average over past period days
    gold_sma = table['gold-price'][t - gold_period:t].mean()
    table.loc[t, 'gold-sma'] = gold_sma

    bitcoin_sma = table['bitcoin-price'][t - bitcoin_period:t].mean()
    table.loc[t, 'bitcoin-sma'] = bitcoin_sma

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
            buy_percentage = bitcoin_buy_amount_multi * sigmoid(buy_bitcoin_indicator)
            # print(buy_percentage)
            buy_value = curr_cash * buy_percentage
            curr_cash, curr_assets = buy(asset='bitcoin',
                                         curr_price=bitcoin_price,
                                         buy_value=buy_value,
                                         cash=curr_cash,
                                         assets=curr_assets,
                                         comission=bitcoin_comission)
        else:
            # to get buy percentage, divide buy bitcoin indicator (same as gradient) by maximum observed gradient and then by 5
            buy_percentage = gold_buy_amount_multi * sigmoid(buy_gold_indicator)
            # print(buy_percentage)
            buy_value = curr_cash * buy_percentage
            curr_cash, curr_assets = buy(asset='gold',
                                         curr_price=gold_price,
                                         buy_value=buy_value,
                                         cash=curr_cash,
                                         assets=curr_assets,
                                         comission=gold_comission)

    if sell_bitcoin_indicator > 0:
        # sell percentage is profit per unit divided by price per unit multiply
        # profit per unit is buying indicator + margin
        sell_percentage = bitcoin_sell_amount_multi * (sell_bitcoin_indicator + bitcoin_margin) / (bitcoin_price)
        sell_amount = curr_assets['bitcoin'] * sell_percentage
        curr_cash, curr_assets = sell(asset='bitcoin',
                                      curr_price=bitcoin_price,
                                      sell_amount=sell_amount,
                                      cash=curr_cash,
                                      assets=curr_assets,
                                      comission=bitcoin_comission)
    if sell_gold_indicator > 0:
        # sell percentage is profit per unit divided by price per unit multiply by 5
        # profit per unit is buying indicator + margin
        sell_percentage = gold_sell_amount_multi * (sell_gold_indicator + gold_margin) / gold_price
        sell_amount = curr_assets['gold'] * sell_percentage
        curr_cash, curr_assets = sell(asset='gold',
                                      curr_price=gold_price,
                                      sell_amount=sell_amount,
                                      cash=curr_cash,
                                      assets=curr_assets,
                                      comission=gold_comission)

    table.loc[t, 'bitcoin-value-at-hand'] = curr_assets['bitcoin'] * bitcoin_price
    table.loc[t, 'gold-value-at-hand'] = curr_assets['gold'] * gold_price
    table.loc[t, 'cash-at-hand'] = curr_cash

    return curr_cash, curr_assets


def simulate(table):
    simulation_results = []
    bitcoin_comission = 0.02
    gold_comission = 0.01

    cash = 10000
    assets = {'bitcoin': 0, 'gold': 0}

    for t in range(len(table)):
        cash, assets = trading_day(t, table, cash, assets, bitcoin_comission, gold_comission)
        if t % 100 == 0:
            day_result = {
                'day': t,
                'cash': int(table['cash-at-hand'][t]),
                'gold_value': int(table['gold-value-at-hand'][t]),
                'bitcoin_value': int(table['bitcoin-value-at-hand'][t])
            }
            simulation_results.append(day_result)
            # print(f"bitcoin_amount:{assets['bitcoin']}")
            # print(f"bitcoin_price:{table['bitcoin-price'][t]}")
            # print(
            #     f"day {t} cash: {table['cash-at-hand'][t]}, gold-value: {table['gold-value-at-hand'][t]}, bitcoin-value: {table['bitcoin-value-at-hand'][t]}")

    # sell anything remaining in the end - not needed anymore
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

    final_value = table['cash-at-hand'][len(table) - 1] + table['gold-value-at-hand'][len(table) - 1] + \
                  table['bitcoin-value-at-hand'][len(table) - 1]
    final_result = {
        'final_value': round(final_value, 2),
        'roi': round(final_value / 100, 2)
    }
    simulation_results.append(final_result)
    return simulation_results
    # print(f"Final value at hand: {round(final_value, 2)} which gives {round(final_value / 100, 2)} % ROI")


def generate_plots(result, static_dir):
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    plot_filenames = []

    stackplot_filename = 'stackplot.png'
    stackplot_path = os.path.join(static_dir, stackplot_filename)
    fig, ax = plt.subplots(figsize=(12, 6))
    stackplot_data = result[['cash-at-hand', 'bitcoin-value-at-hand', 'gold-value-at-hand']]
    stackplot_data.plot.area(ax=ax, colormap='Paired')
    fig.savefig(stackplot_path)
    plt.close(fig)  # Close the figure to free memory
    plot_filenames.append(stackplot_filename)

    # Plot for Bitcoin Buy Amount
    result['bitcoin_buy_amount'] = 0.2 * result[['buy_bitcoin_indicator']].apply(sigmoid)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=result[['bitcoin_buy_amount']], ax=ax)
    bitcoin_buy_amount_filename = 'bitcoin_buy_amount.png'
    fig.savefig(os.path.join(static_dir, bitcoin_buy_amount_filename))
    plt.close(fig)
    plot_filenames.append(bitcoin_buy_amount_filename)

    # Plot for Gold Buy Amount
    result['gold_buy_amount'] = 0.2 * result[['buy_gold_indicator']].apply(sigmoid)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=result[['gold_buy_amount']], ax=ax)
    gold_buy_amount_filename = 'gold_buy_amount.png'
    fig.savefig(os.path.join(static_dir, gold_buy_amount_filename))
    plt.close(fig)
    plot_filenames.append(gold_buy_amount_filename)

    # Plot for Gold and Bitcoin Prices with Buy Dates
    buy_gold_days = list(result.loc[result['buy_bitcoin_indicator'] > 0]['t'])
    buy_bitcoin_days = list(result.loc[result['buy_gold_indicator'] > 0]['t'])
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title("Gold and Bitcoin Prices with Suggested Buy Dates")
    for day in buy_gold_days:
        ax.axvline(day, color='black')
    for day in buy_bitcoin_days:
        ax.axvline(day, color='r')
    sns.lineplot(data=result[['bitcoin-price', 'gold-price']], ax=ax)
    prices_filename = 'gold_bitcoin_prices.png'
    fig.savefig(os.path.join(static_dir, prices_filename))
    plt.close(fig)
    plot_filenames.append(prices_filename)

    return plot_filenames


def run_simulation(bitcoin_file, gold_file, static_dir):
    result = load_data(bitcoin_file, gold_file)
    simulate(result)
    generate_plots(result, static_dir)

if __name__ == '__main__':
    result = load_data()
    simulate(result)
    generate_plots(result)
