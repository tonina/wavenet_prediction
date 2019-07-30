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


class DataBatch(object):
    '''
    Class for generation data batches from dataset.
    '''
    def __init__(self, company_name, start_date, end_date, deep, batch_size,
                 source='yahoo', start_i=0, stop_i=None):
        self.batch_size = batch_size
        self.deep = deep
        self.data = load_company_data(company_name, start_date, end_date, source)
        self.adj_close = self.data['Adj Close'].values
        self.i = start_i
        self.stop_i = len(self.data.index) - deep - 1

    def __iter__(self):
        return(self)

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


if __name__ == '__main__':
    start = datetime(2019, 7, 1)
    end = datetime.today()
    d_batch = DataBatch('WMT', start, end, 5, batch_size=8)
    while True:
        try:
            x, t = next(d_batch)
            print('x', x.shape)
            print('t', t.shape)
        except StopIteration:
            break
