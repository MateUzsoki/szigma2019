# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:17:17 2018

@author: Uzsoki.Mate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import os

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import math
import constants

yf.pdr_override()
def get_tickers():
    path="..\Data"
    allFiles = glob.glob(path + "/*.csv")
    return allFiles
    
def get_returns_for_single_stock(file_,start_date):
    
    df = pd.read_csv(file_,index_col="Date", header=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    mask = (df.index >= start_date)
    df = df.loc[mask]
    #file_name = os.path.splitext(os.path.basename(file_))[0]
    #df["Stock"] = file_name
    df["Return"] = (df["Adj Close"] - df["Adj Close"].shift(1)) / df["Adj Close"].shift(1)
    df = df.dropna(axis=0,how='all')
    df = df.fillna(0)
    count = df.count()[0]
    if(count < 7560):
        print("!!!!!!!")
        print(file_)
        print(count)
        
    if(count > 7561):
        print("!!!!!!!")
        print(file_)
        print(count)
    #print(df.index.max())
    #print(df.index.min())
    return df

def get_returns(start_date,end_date, tickers):
    #allFiles = get_tickers()
    
    #frame = pd.DataFrame()
    #list_ = []
    #for file_ in allFiles:
    #    df = get_returns_for_single_stock(file_)
    #    list_.append(df)
    #frame = pd.concat(list_)

    panel_data = pdr.get_data_yahoo(tickers, start=start_date, end=end_date)
    
    #panel_data = pdr.DataReader(tickers,data_source,start_date,end_date)
    
    #selecting closing prices
    close = panel_data['Close']
    
    # Getting all weekdays 
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # How do we align the existing prices in adj_close with our new set of dates?
    # All we need to do is reindex close using all_weekdays as the new index
    close = close.reindex(all_weekdays)

    returns = pd.DataFrame()
    for sym in tickers:
            returns[sym] = (close[sym] - close[sym].shift(1) ) / (close[sym])
    
    #drop all NaN rows
    returns = returns.dropna(axis=0,how='all')

    #fill in NaNs with zero
    returns = returns.fillna(0)

    return returns

def split_into_rows_no_overlap(singleStock,N):
    tr = pd.DataFrame()
    row = 0
    
    for x in range(0,(singleStock.count() // N)*N ):
        row = x // N
        column = x - (row * N) 
        tr.at[row,column] = singleStock[x]
       # tr.set_value(row, column, singleStock[x])
    return tr

def split_into_rows_overlap(singleStock,N):
    singleStockReturn = pd.DataFrame()
      #egymást átfedő időszakok, egy napos eltolás
    for x in range(1, N + 1):
        singleStockReturn[x] = singleStock.shift(x)
    singleStockReturn = singleStockReturn.dropna(axis=0,how='any')
    return singleStockReturn
    
def get_esreal(N,confLevel,tr):
    trt = tr.transpose()
    ESReal3 = pd.DataFrame()
    itemNr = N*(1-confLevel)
    k1 = int(math.floor(itemNr)) # lefelé kerekít
    k2 = int(math.ceil(itemNr)) # felfelé kerekít
    weights = np.append(np.ones(k1),itemNr-k1)
    
    for x in range(0,len(trt.columns)) :# nullától indul , az első elemet el kell dobni ellenőrzésnél
        es2 = trt.nsmallest(k2,trt.columns[x])[trt.columns[x]]
        es2weighted = es2*weights
        ESReal3.set_value(index=x,col= 0 ,value=es2weighted.sum()/itemNr*-1 )
 
    return ESReal3
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def get_stockname(selected):
    return selected.split(".")[2].split('\\')[2]

def get_all_stock_dataset(N,confLevel,rollingWindow,test_size,start_date):
    
    start_date = constants.start_date
    N = constants.N
    confLevel = constants.confLevel
    test_size=constants.test_size
    rollingWindow = constants.rollingWindow
    all_stocks_train = []
    all_stocks_test_with_period = pd.DataFrame(columns = ['Stock', 'Period' , 'HS_pred','ANN_real','ANN_pred'])
    
    all_stocks_test = []
    tickers = get_tickers()    
    for selected in tickers:
        print(selected)
        singleStock = get_returns_for_single_stock(selected,start_date)
        tr = split_into_rows_no_overlap(singleStock["Return"],N)
        
        tr_unscaled = pd.DataFrame(get_technical_indicies(N,confLevel,tr,rollingWindow))
        tr_unscaled["StockName"] = selected
        splitIndex = int(tr_unscaled.shape[0]*test_size)
        tr_train = tr_unscaled.iloc[:splitIndex,:]
        tr_test = tr_unscaled.iloc[splitIndex:,:]
        all_stocks_train.append(tr_train)
        all_stocks_test.append(tr_test)
        
        tr_test.iloc[0,1]
        for ii in range(0,len(tr_test.columns)+1):
             peaIndex = get_stockname(selected)+"_"+ str(tr_test.index.values[ii]);
             all_stocks_test_with_period.loc[peaIndex] = [get_stockname(selected),tr_test.index.values[ii],tr_test.iloc[ii,0],tr_test.iloc[ii,11] ,0]    
        
    all_stocks_train = pd.concat(all_stocks_train)
    all_stocks_train_np = all_stocks_train.values
    labelencoder = LabelEncoder()
    
    all_stocks_train_np[:,-1] = labelencoder.fit_transform(all_stocks_train_np[:,-1])
    
    onehotencoder = OneHotEncoder(categorical_features = [all_stocks_train_np.shape[1]-1])
    all_stocks_train_np = onehotencoder.fit_transform(all_stocks_train_np).toarray()
    all_stocks_train_np = all_stocks_train_np[:,1:]
    
    all_stocks_test = pd.concat(all_stocks_test)
    all_stocks_test_np = all_stocks_test.values
    all_stocks_test_np = all_stocks_test.values
    all_stocks_test_np[:,-1] = labelencoder.transform(all_stocks_test_np[:,-1])    
    all_stocks_test_np = onehotencoder.transform(all_stocks_test_np).toarray()
    all_stocks_test_np = all_stocks_test_np[:,1:]
    
    print("all_stocks_test_np :" ,all_stocks_test_np.shape)
    print("all_stocks_train_np :" ,all_stocks_train_np.shape)
    
    return all_stocks_train_np, all_stocks_test_np, all_stocks_test_with_period

def get_technical_indicies(N,confLevel,tr,rollingWindow):
    ESReal = get_esreal(N,confLevel,tr)
    tr_ext = ESReal
    tr_ext = tr_ext.drop([0], axis = 1)
    
  
    tr_ext["ES"] = ESReal.values
    tr_ext["VaR"] = tr.quantile(q=confLevel, axis=1).values
    tr_ext["mean"] = tr.mean(axis=1).values
    tr_ext["var"] = tr.var(axis=1).values

    
    middle_band = tr_ext["ES"].rolling(rollingWindow,rollingWindow).mean()
    lower_band = middle_band - 2 * tr_ext["ES"].rolling(rollingWindow,rollingWindow).std()
    upper_band = middle_band + 2 * tr_ext["ES"].rolling(rollingWindow,rollingWindow).std()
    momentum1 = tr_ext["ES"] - tr_ext["ES"].shift(math.floor(rollingWindow/2))
    momentum2 = tr_ext["ES"] - tr_ext["ES"].shift(math.floor(rollingWindow/2)*2)
     
    ewma = pd.ewma(arg=tr_ext["ES"], span=rollingWindow)
    
    tr_ext["middle_band"] = middle_band
    tr_ext["lower_band"] = lower_band
    tr_ext["upper_band"] = upper_band
    
    tr_ext["momentum1"] = momentum1 
    tr_ext["momentum2"] = momentum2 
    tr_ext["acceleration"] = momentum1- momentum2 
    tr_ext["EWMA"] =ewma 
    
        
    tr_ext = tr_ext.dropna(axis=0,how='any')
    
    tr_ext_before_y = tr_ext.iloc[:-1] #eldobjuk az utolsó sort 
    
    es_real_final =ESReal[rollingWindow+1:].values 
   
    tr_ext_with_y = tr_ext_before_y 
    tr_ext_with_y["y"] = es_real_final
    
    return tr_ext_with_y
    