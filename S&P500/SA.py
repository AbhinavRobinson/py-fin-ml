# Stock Analysis

from collections import Counter
import numpy as np
import pandas as pd
import pickle

hm_days = 7


def process_data_for_labels(ticker):
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (
            df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)

    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    # 2% fluctuation
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_feature_sets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(
        map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)]for i in range(1, hm_days+1)]))
    str_vals = [str(i) for i in df['{}_target'.format(ticker)].values.tolist()]
    print('Data Spread:', Counter(str_vals))
