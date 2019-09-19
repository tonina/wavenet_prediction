from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data


def load_company_data(company_name, start_date, end_date, source):
    '''
    Load company stock data from finance.yahoo.com by company name.
    :param source:
    :param end_date:
    :param start_date:
    :param company_name: str
    :return: pandas dataframe - with time series of company stock data
    '''
    start_d = start_date.strftime('%Y-%m-%d')
    end_d = end_date.strftime('%Y-%m-%d')
    hist_data = data.DataReader(company_name,
                                start=start_d,
                                end=end_d,
                                data_source=source)
    return hist_data


class DataTestBatch(object):
    '''
    Class for generation data batches from dataset.
    Create inputs and target output sets in form {input series} - {target next value}
    Shifted by one value. Suitable for autoregressive network.
    '''
    def __init__(self, company_name, start_date, end_date, deep, batch_size,
                 source='yahoo', start_i=0, stop_i=None):
        self.batch_size = batch_size
        self.deep = deep
        self.data = load_company_data(company_name, start_date, end_date, source)
        self.adj_close = self.data['Adj Close'].values
        self.i = start_i
        self.stop_i = len(self.data.index) - deep - 1 if stop_i is None else stop_i

    def __iter__(self):
        return self

    def __next__(self):
        if self.i > self.stop_i:
            raise StopIteration
        else:
            ins = []
            tars = []
            for k in range(self.batch_size):
                try:
                    tars.append(self.adj_close[self.i+k+self.deep])
                    ins.append(self.adj_close[self.i+k: self.i+k+self.deep])
                except IndexError:
                    pass
            self.i += self.batch_size
            X = np.array(ins)
            T = np.array(tars)
            T = np.expand_dims(T, axis=1)
            return X, T


class DataTrainBatch(object):
    '''
    Class for generation data batches from dataset.
    Create only inputs data, for convolutional network.
    '''
    def __init__(self, company_name, start_date, end_date, batch_size, deep, deep_factor, step,
                 source='yahoo', start_i=0, stop_i=None):
        self.batch_size = batch_size
        self.length = int(deep * deep_factor)
        self.step = step
        self.data = load_company_data(company_name, start_date, end_date, source)
        self.adj_close = self.data['Adj Close'].values
        self.i = start_i
        self.stop_i = len(self.data.index) - self.length if stop_i is None else stop_i

    def __iter__(self):
        return self

    def __next__(self):
        if self.i > self.stop_i:
            raise StopIteration
        else:
            ins = []
            for k in range(self.batch_size):
                try:
                    item = self.adj_close[self.i+k: self.i+k+self.length]
                    if len(item) == self.length:
                        ins.append(item)
                except IndexError:
                    pass
            self.i += self.batch_size
            X = np.array(ins)
            return X


if __name__ == '__main__':
    start = datetime(2019, 7, 1)
    end = datetime.today()
    d_batch = DataTrainBatch('WMT', start, end, 6, 20, 1.6, 2)
    while True:
        try:
            x = next(d_batch)
            print('x', x.shape)

        except StopIteration:
            break
