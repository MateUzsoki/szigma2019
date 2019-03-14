# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 07:59:33 2018

@author: Uzsoki.Mate
"""

import datetime

start_date = datetime.datetime(1988,1,1)
end_date = datetime.datetime(2018,1,1)
#tickers = ['AAPL','AXP','BA','CAT','CVX','GE','IBM','JNJ','JPM','KO','MCD','MSFT','PFE','PG','TRV']
#tickers = ['CAT','CVX']
N = 250
MC = 10000
confLevel = 0.975 #confidence level
test_size=0.5
number_of_epochs = 200
rollingWindow = 2 # calculating ANN technical indexes results in 2 less testable data
M2Correction = 1 # calculating Momentum 2 results in 1 less testable data

periods = (7560 // N ) 

# the ratio of the test size (14) to the total number of testable periods in N-MC (29)
corrected_test_size = ( (periods*test_size - rollingWindow+1) / ( (periods-1)*test_size) )*test_size

loss_function = "mean_absolute_error"