import os
import sys
import requests
import datetime as dt

import numpy as np
import pandas as pd
import unicodedata
from unidecode import unidecode

def unicode2float(number_unicode):
    """
    Convert unicode text to float number

    Args:
        number_unicode: a number as unicode text type

    Returns:
        the float number
    """
    if number_unicode == '': return 0
    number_str = number_unicode.encode('ascii', 'ignore')
    number_text = number_str.replace(",","")
    number_float = float(number_text)
    return number_float

def normal_symbol(symbol, env):
    """
    Check a symbol is normal or not, it means not index, not foregin symbols

    Args:
        symbol: symbol to check
        env: environment variable

    Return:
        True if it is normal symbol else False
    """
    if symbol[0] != '^' and symbol not in env['blacklist_symbols']:
        return True
    return False

def viet2plain(text):
    """
    Convert vietnamese to plain text, i.e, 'Nguy\xean' -> 'Nguyen'

    Args:
        text: vietnamese

    Return:
        plain text
    """
    # Normalize then encode has some problem with 1-char characters (when
    # parse from internet), try it, if fail then should use unidecode
    # xtuanta: I found that this way is very good way to do this convertation
    return ''.join([unidecode(c).lower() for c in text])
    # try:
    #     text = unicode(text, 'utf-8')
    # except: # unicode is a default on python 3
    #     pass
    # text = unicodedata.normalize('NFD', text)
    # text = text.encode('ascii', 'ignore')
    # text = text.decode("utf-8")
    # return str(text)
    # try:
    #     ascii_name = unidecode(text)
    # except UnicodeDecodeError as e:
    #     norm_name = unicodedata.normalize('NFKD', text)
    #     ascii_name = norm_name.encode('ASCII', 'ignore')
    #
    # return ascii_name

def try_request(url, method='post', max_retries=5, session=None, cookies=None,
                data=None, **kwargs):
    if session is None:
        request = requests.post if method is 'post' else requests.get
    else:
        request = session.post if method is 'post' else session.get
    #
    for i in range(max_retries+1):
        if i == max_retries: return False
        try:
            res = request(url, cookies=cookies, data=data, **kwargs)
            break
        except:
            print ("Unexpected error:", sys.exc_info())
            continue
        #
    #
    return res

def load_symbols(env):
    #filename = env['data_dir'] + 'symbols.txt'
    filename = env["working_dir"] + 'symbols.txt'    
    if not os.path.exists(filename):
        env = init_environment(symbols=None)
        from stock import Cophieu68
        cophieu68 = Cophieu68(env)
        symbols = cophieu68.get_symbols()
        if symbols is False: return None, None
    # THAY DOI MOT SO DUONG DAN
    # JUST EXPERIMENTING FOR A BIT
    #f = open(env['data_dir'] + 'symbols.txt', "r")
    f = open(env['working_dir']+'symbols.txt','r')
    symbols = f.read().splitlines()
    f.close()
    return symbols

def compute_return(df, column):
    return np.log(df[column] / df[column].shift(1))


"""
Setting the min/max target for our data
"""
def set_targets(dfs, stocks_df=None):
    for symbol in dfs['symbol'].unique():
        df = dfs[dfs['symbol'] == symbol]
        if stocks_df is None:
            stock_df = load_stock(symbol)
        else:
            stock_df = stocks_df[stocks_df['symbol'] == symbol]
        #
        for index, row in df.iterrows():
            print( symbol, row['time'])
            curr_date = dt.date(row['year'], row['quarter']*3, 1)
            if curr_date < stock_df['date'].min().to_datetime().date(): continue
            time_range = 3 * 252 # days
            future_df = stock_df[stock_df['date'] > curr_date].head(time_range)# 3 years, now I'm using 1 year
            if future_df.shape[0] == 0: continue
            # min/max target
            max_target = future_df['close'].max()
            min_target = future_df['close'].min()
            # expected target, a weight combined between probability of this
            # price (histogram) and a straight line `weights`
            weights = np.array([i for i in range(future_df.shape[0])]) + 1
            weights = weights / float(np.sum(weights))
            counts, bins = np.histogram(future_df['close'], bins='auto')
            probs = counts / float(np.sum(counts))
            future_df['prob'] = future_df['close'].apply(lambda x:
                probs[max(np.where(bins >= x)[0][0] - 1, 0)])
            future_df['weight'] = future_df['prob'] * weights
            future_df['weight'] = future_df['weight'] / future_df['weight'].sum()
            expected_target = np.average(future_df['close'],
                weights=future_df['weight'])
            #
            dfs = dfs.set_value(index, 'max_target', max_target)
            dfs = dfs.set_value(index, 'min_target', min_target)
            dfs = dfs.set_value(index, 'expected_target', expected_target)
    #
    return dfs

def init_environment(**kwargs):
    env = dict()
    for key, value in kwargs.iteritems():
        env[key] = value
    #
    if 'working_dir' not in env:
        #working_dir = os.path.dirname(os.path.realpath('__file__')) + '/'
        #working_dir = os.path.dirname(os.path.realpath('')) + '/'
        working_dir = os.path.realpath('')+'/'        
        env['working_dir'] = working_dir
    if 'data_dir' not in env: env['data_dir'] = env['working_dir'] + 'data/'
    if 'tmp_dir' not in env: env['tmp_dir'] = env['data_dir'] + 'tmp/'
    if 'web_dir' not in env: env['web_dir'] = env['data_dir'] + 'web/'
    for key in env.keys():
        if '_dir' not in key: continue
        folder = env[key]
        if os.path.isdir(folder): continue
        os.makedirs(folder)
    #
    # TODO(xtuanta): need Calendar class here
    today = dt.date.today()
    env['start_date'] = pd.Timestamp('2000-07-28')
    env['special_nontrading_days'] = [
        pd.Timestamp('2008-05-27'), # technical issue
        pd.Timestamp('2008-05-28'), # technical issue
        pd.Timestamp('2008-05-29'), # technical issue
        #
        pd.Timestamp('2014-09-01'), # National Day
        pd.Timestamp('2014-09-02'), # National Day
        #
        pd.Timestamp('2015-01-01'), # New Year
        pd.Timestamp('2015-01-02'), # New Year
        pd.Timestamp('2015-02-16'), # Lunar New Year
        pd.Timestamp('2015-02-17'), # Lunar New Year
        pd.Timestamp('2015-02-18'), # Lunar New Year
        pd.Timestamp('2015-02-19'), # Lunar New Year
        pd.Timestamp('2015-02-20'), # Lunar New Year
        pd.Timestamp('2015-02-21'), # Lunar New Year
        pd.Timestamp('2015-02-22'), # Lunar New Year
        pd.Timestamp('2015-02-23'), # Lunar New Year
        pd.Timestamp('2015-04-28'), # Liberation Day and Labor Day
        pd.Timestamp('2015-04-29'), # Liberation Day and Labor Day
        pd.Timestamp('2015-04-30'), # Liberation Day and Labor Day
        pd.Timestamp('2015-05-01'), # Liberation Day and Labor Day
        pd.Timestamp('2015-09-02'), # National Day
        #
        pd.Timestamp('2016-01-01'), # New Year
        pd.Timestamp('2016-02-08'), # Lunar New Year
        pd.Timestamp('2016-02-09'), # Lunar New Year
        pd.Timestamp('2016-02-10'), # Lunar New Year
        pd.Timestamp('2016-02-11'), # Lunar New Year
        pd.Timestamp('2016-02-12'), # Lunar New Year
        pd.Timestamp('2016-04-18'), # Hung King's Day
        pd.Timestamp('2016-05-02'), # Liberation Day and Labor Day
        pd.Timestamp('2016-05-03'), # Liberation Day and Labor Day
        pd.Timestamp('2016-09-02'), # National Day
        #
        pd.Timestamp('2017-01-02'), # New Year
        pd.Timestamp('2017-01-26'), # Lunar New Year
        pd.Timestamp('2017-01-27'), # Lunar New Year
        pd.Timestamp('2017-01-28'), # Lunar New Year
        pd.Timestamp('2017-01-29'), # Lunar New Year
        pd.Timestamp('2017-01-30'), # Lunar New Year
        pd.Timestamp('2017-01-31'), # Lunar New Year
        pd.Timestamp('2017-02-01'), # Lunar New Year
        pd.Timestamp('2017-04-06'), # Hung King's Day
        pd.Timestamp('2017-05-01'), # Liberation Day and Labor Day
        pd.Timestamp('2017-05-02'), # Liberation Day and Labor Day
        pd.Timestamp('2017-09-04'), # National Day
    ]
    env['nontrading_days'] = env['special_nontrading_days']
    for day in pd.date_range(env['start_date'], today):
        if day.weekday() < 5: continue
        if day in env['nontrading_days']: continue
        env['nontrading_days'].append(day)
    #
    all_day = pd.date_range(env['start_date'], today)
    env['trading_days'] = list(set(all_day.tolist()) - set(env['nontrading_days']))
    #
    env['blacklist_symbols'] = ['000001.ss']
    #
    if 'symbols' not in env:
        symbols = load_symbols(env)
        env['symbols'] = symbols
    # if 'fund' not in env:
    #     from fundamental import Fundamental
    #     env['fund'] = Fundamental(env)
    # if 'metadatas' not in env:
    #     env['metadatas'] = env['fund'].load_metadatas()
    if 'stocks_df' not in env:
        filename = env['data_dir'] + 'stocks.csv'
        if os.path.exists(filename):
            env['stocks_df'] = pd.read_csv(filename, parse_dates=['date'])
        else:
            env['stocks_df'] = None
    if 'init' not in env: env['init'] = 100e6 / 1000 #unit 1000 vnd
    if 'tax' not in env: env['tax'] = 0.1 / 100
    if 'fee' not in env: env['fee'] = 0.15 / 100
    if 'portfolio' not in env:
        #
        # from portfolio import Portfolio
        # portfolios = Portfolio(env)
        # env['portfolio'] = portfolios.get_portfolios()
        #
    #
        return env
