import requests
import json
import pandas as pd
import datetime as dt
from config import API_SECRET,API_KEY
from binance import Client
from get_data_binance import getbinancedaily
from get_data_bitstamp import getbitstampdaily
from math import *
import numpy as np
from ts2vg import NaturalVG
import networkx as nx
import pickle as pkl
from ta.trend import ADXIndicator
import os
import argparse

# Functions

def SMA(df, period, price, delta):
    SMA = price.rolling(period, min_periods=1, closed='both').mean()
    return SMA


def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

def RSI(df, period, price, delta):
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    
    return rsi+0.1

def EMA(df, period, price, delta):
    SMA = price.rolling(period, min_periods=1, closed='both').mean()
    mult = 2/(period+1)
    EMA = [SMA[0]]
    for i in range(len(SMA[1:])):
        EMA.append((price[i+1]-EMA[i])*mult+EMA[i])
    return EMA

def volatility(df, period, price, delta):
    std = price.rolling(period, min_periods=1, closed='both').std()
    Vol = std * sqrt(period)
    return Vol

def shortest_path_length(close: np.array, lookback: int):

    avg_short_dist_p = np.zeros(len(close))
    avg_short_dist_n = np.zeros(len(close))

    avg_short_dist_p[:] = np.nan
    avg_short_dist_n[:] = np.nan

    for i in range(lookback, len(close)):
        dat = close[i - lookback + 1: i+1]

        pos = NaturalVG()
        pos.build(dat)

        neg = NaturalVG()
        neg.build(-dat)

        neg = neg.as_networkx()
        pos = pos.as_networkx()
        avg_short_dist_p[i] = nx.average_shortest_path_length(pos)
        avg_short_dist_n[i] = nx.average_shortest_path_length(neg)
    return avg_short_dist_p, avg_short_dist_n

def Bollinger(df, period, price, delta):
    Mid = SMA(df, period, price, delta)
    Delta = volatility(df, period, price, delta)/sqrt(period)
    Top = Mid + 2*Delta
    Bot = Mid - 2*Delta
    return Top, Bot

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default= 'AVAX')
    args = parser.parse_args()
    ticker = args.ticker
    dfBn = getbinancedaily(sym=f'{ticker}USDT')
    dfBt = getbitstampdaily(sym=f'{ticker.lower()}usd')
    dfBn.to_csv('data-binance.csv', index = False)
    dfBt.to_csv('data-bitstamp.csv', index = False)
    dfBn = pd.read_csv('data-binance.csv')
    dfBt = pd.read_csv('data-bitstamp.csv')
    df = pd.DataFrame()

    df['open'] = (dfBn['open']*dfBn['volume']+dfBt['open']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['high'] = (dfBn['high']*dfBn['volume']+dfBt['high']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['low'] = (dfBn['low']*dfBn['volume']+dfBt['low']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    df['close'] = (dfBn['close']*dfBn['volume']+dfBt['close']*dfBt['volume'])/(dfBt['volume']+dfBn['volume'])
    delta = df['close'].diff()
    price = df['close']
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    price_open = df['open'].values
    price_high = high
    price_low = low
    ADX = ADXIndicator(df['high'], df['low'], df['close'], window = 14, fillna = True)
    df['ADX'] = ADX.adx()


    df['SMA_50'] = SMA(df, 50, price, delta)
    df['EMA_100'] = EMA(df, 100, price, delta)
    df['volume'] = dfBn['volume']+dfBt['volume']
    df['vol'] = volatility(df, 14, price, delta)
    df['RSI_14'] = RSI(df, 14, price, delta)

   
    print(df.head)
        
    # r = requests.get(url = 'https://api.alternative.me/fng/', params = {'limit': 0, 'date_format': ''})
    # r = r.json()['data']
    # dfG = pd.DataFrame(r)
    # dfG =dfG.iloc[::-1]

    # dfG['timestamp'].values
    # dfG.index = [dt.datetime.fromtimestamp(int(x)) for x in dfG.timestamp]
    # L = dfG['timestamp'].values
    # L = [int(a) for a in L]
    # L = list(range(L[0],L[-1]+1, 24*3600))
    # for i,a in enumerate(L) :
    #     L[i] = dt.datetime.fromtimestamp(a)
    # dfG = dfG.reindex(L,method="ffill")
    # print(dfG.index)


    df['date'] = dfBn['date']


    # df['Boll_Top'], df['Boll_Bot'] = Bollinger(df, 24, price, delta)
    df = df.dropna()
    # df = df.iloc[-(len(dfG.index)-27):]
    # df['sentiment'] = dfG['value'].values[:-27]
    # print(dfG['value'])

    # cols2 = ['date', 'price', 'price_high', 'price_low', 'price_open', 'close']
    # df2 = df[cols2]

    cols = ['date', 'volume', 'vol', 'RSI_14', 'EMA_100', 'SMA_50', 'ADX', 'open', 'high', 'low', 'close']
    df = df[cols]


    df.to_parquet(f'Data_{ticker}_Daily.parquet')
    os.remove('data-binance.csv')
    os.remove('data-bitstamp.csv')
    # df.to_csv("/home/owen/Documents/NeurIPS2023-One-Fits-All/Long-term_Forecasting/datasets/BTC/BTC_Daily_fracdiffnodata.csv", index=False)
    # df.to_csv("/home/owen/Documents/NeurIPS2023-One-Fits-All/Long-term_Forecasting/datasets/BTC/BTC_DailyfracdiffADF_price.csv", index=False)
    # df.to_csv("/home/owen/Documents/NeurIPS2023-One-Fits-All/Long-term_Forecasting/datasets/BTC/BTC_Daily_sentiment.csv", index=False)
    # df2.to_csv("/home/owen/Documents/NeurIPS2023-One-Fits-All/Long-term_Forecasting/datasets/BTC/BTC_DailyfracdiffADF.csv", index=False)
    



