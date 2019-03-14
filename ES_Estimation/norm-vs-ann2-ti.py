# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 07:48:24 2018

@author: Uzsoki.Mate
"""


import numpy as np
import pandas as pd
import datetime
import math
import constants
import muz_lib 

tickers = muz_lib.get_tickers()


#loading data
#stock_returns = muz_lib.get_returns(start_date,end_date,tickers)

estimationAccuracy = pd.DataFrame(columns = ['Stock', 'Method' , 'MAE', 'MRE', 'MSE'])
periodicEstimationAccuracy = pd.DataFrame(columns = ['Stock', 'Period' , 'Real1', 'PredNMC'])

selected = 'ALL'  
#
totalTestSize = 0
totalTrainSize = 0


from matplotlib import pyplot

for selected in tickers:
    print(selected)

    singleStock = muz_lib.get_returns_for_single_stock(selected,constants.start_date)["Return"]

    #tr = muz_lib.split_into_rows_overlap(singleStock,N)   
    singleStockReturn = muz_lib.split_into_rows_no_overlap(singleStock,constants.N)
       
    #minden időszakhoz számolunk szórást és átlagot
    parameters = pd.DataFrame()
    parameters["std"] = singleStockReturn.std(axis=1)
    parameters["mean"] = singleStockReturn.mean(axis=1)
    
    #
    
    ESSim = pd.DataFrame()
    k = int(round(constants.MC*(1-constants.confLevel)))
    
    for i in range(0 ,len(parameters) -1 ):  #-1 , mert az utolsó nem kell, azt úgy sem lehet ellenőrizni
         #if(i/ (len(parameters) -1)>=(1-test_size)):        
         simulated =  np.random.normal(parameters.iloc[i]["mean"],parameters.iloc[i]["std"],constants.MC)    
         idx = np.argpartition(simulated, k)
         nsmallest = simulated[idx[:k]]
         es = np.mean(nsmallest)*-1
         ESSim.set_value(index=i,col= 0 ,value=es)
   
    
    ESReal = muz_lib.get_esreal(constants.N,constants.confLevel,singleStockReturn)
        
        
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    
    y_test = ESReal[1:].values
    y_pred = ESSim
    
   
    
    #itt az eredmények egy részét eldobjuk, hogy ugyanazon az időszakon teszteljünk, mint az ANN-nél
    from sklearn.model_selection import train_test_split
    real_train, real_test, pred_train,pred_test = train_test_split(ESReal[1:].values,ESSim,test_size=constants.corrected_test_size,shuffle=False)
    
    totalTrainSize = totalTrainSize + real_train.size
    totalTestSize = totalTestSize + pred_test.size
    
    evs = explained_variance_score(real_test,pred_test)
    mae = mean_absolute_error(real_test,pred_test)
    mre = abs(mae / real_test.mean())
    mse = mean_squared_error(real_test,pred_test)
#    print("evs:",evs ,"mae:" ,mae,"mse:",mse, "mre:", mre)
    
    row = [selected,'Norm',mae,mre,mse]
    
    for ii in range(0 ,len(pred_test)):
        peaIndex = muz_lib.get_stockname(selected)+"_"+ str(pred_test.index.values[ii]);
        periodicEstimationAccuracy.loc[peaIndex] = [ muz_lib.get_stockname(selected),pred_test.index.values[ii],real_test[ii,0],pred_test.iloc[ii,0]]


#*****************************************
#  ANN 
#*****************************************

from sklearn.preprocessing import MinMaxScaler



all_stocks_train_np, all_stocks_test_np, all_stocks_test_no_label_encode = muz_lib.get_all_stock_dataset(constants.N,constants.confLevel,constants.rollingWindow,1-constants.test_size,constants.start_date)
    
    
#X = training_set_unscaled[:,:-1]
#y = training_set_unscaled[:,-1:]
    
#from sklearn.model_selection import train_test_split
#X_train_unscaled, X_test_unscaled, y_train_unscaled,y_test_unscaled = train_test_split(X,y,test_size=test_size,shuffle=True)

X_train_unscaled = all_stocks_train_np[:,:-1]
y_train_unscaled = all_stocks_train_np[:,-1:]
X_test_unscaled = all_stocks_test_np[:,:-1]
y_test_unscaled = all_stocks_test_np[:,-1:]

from sklearn.preprocessing import MinMaxScaler
scX = MinMaxScaler(feature_range = (0,1))
X_train_scaled = np.array(scX.fit_transform(X_train_unscaled))
X_test_scaled = np.array(scX.transform(X_test_unscaled))

scy = MinMaxScaler(feature_range = (0,1))
y_train_scaled = np.array(scy.fit_transform(y_train_unscaled))
y_test_scaled = np.array(scy.transform(y_test_unscaled))

from sklearn.preprocessing import StandardScaler

#    X_train = sc.fit_transform(X_train)
#    X_test = sc.transform(X_test)

##ANN 
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

  


#evaluation the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


classifier = Sequential()

#Hidden layer , activation funciton = rectifier

classifier.add(Dense(units = X_train_scaled.shape[1],kernel_initializer= "uniform", activation = 'sigmoid', input_dim= X_train_scaled.shape[1]))
   # classifier.add(Dropout(0.1))

#hidden 2
classifier.add(Dense(units = X_train_scaled.shape[1],kernel_initializer= "uniform", activation = 'sigmoid'))

   # classifier.add(Dropout(0.1))
#hidden 3
   # classifier.add(Dense(units = X.shape[1],kernel_initializer= "uniform", activation = 'sigmoid'))

#output layer -> just one output node
#sigmoid for probabilistic approach ## softmax instead of sigmoid if more than two categories
classifier.add(Dense(units = 1,kernel_initializer= "uniform", activation = 'sigmoid'))

#opt = stochactic gradient ('adam')
#loss = logarithmic loss -> more than one output "categorical_crossentropy"
classifier.compile(optimizer = "nadam",loss=constants.loss_function, metrics= ['mae'])

classifier.fit(X_train_scaled,y_train_scaled,batch_size=10,nb_epoch = constants.number_of_epochs)
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
y_pred_scaled = classifier.predict(X_test_scaled)


y_pred_unscaled=scy.inverse_transform(y_pred_scaled)[:,-1:]

all_stocks_test_no_label_encode["ANN_pred"]  = y_pred_unscaled

combinedEstimationAccuracy = pd.merge(all_stocks_test_no_label_encode, periodicEstimationAccuracy, left_index=True, right_index=True)
    
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#evs = explained_variance_score(y_test,y_pred)
mae = mean_absolute_error(y_test_unscaled,y_pred_unscaled)
mre = abs(mae / y_test_unscaled.mean())
mse = mean_squared_error(y_test_unscaled,y_pred_unscaled)

estimationAccuracy.loc[estimationAccuracy["Stock"].count()] =[selected,'ANN',mae,mre,mse]



y1 = y_test_unscaled[1:]
y2=y_test_unscaled[:-1]


mae2 = mean_absolute_error(y1,y2)
mre2 = abs(mae2 / y1.mean())
mse2 = mean_squared_error(y1,y2)
estimationAccuracy.loc[estimationAccuracy["Stock"].count()] =[selected,'Hist',mae2,mre2,mse2]

print("--------------------------------------------------------------------")
print("ANN MSE:" ,estimationAccuracy.loc[estimationAccuracy["Method"] == "ANN"]["MSE"].mean()*100000)
print("Hist MSE:" ,estimationAccuracy.loc[estimationAccuracy["Method"] == "Hist"]["MSE"].mean()*100000)
print("Norm MSE:" ,estimationAccuracy.loc[estimationAccuracy["Method"] == "Norm"]["MSE"].mean()*100000)

print("ANN MAE:" ,estimationAccuracy.loc[estimationAccuracy["Method"] == "ANN"]["MAE"].mean()*1000)
print("Hist MAE:" ,estimationAccuracy.loc[estimationAccuracy["Method"] == "Hist"]["MAE"].mean()*1000)
print("Norm MAE:" ,estimationAccuracy.loc[estimationAccuracy["Method"] == "Norm"]["MAE"].mean()*1000)
print("N:",constants.N," conflevel: ",constants.confLevel," testsize:",constants.test_size, " min-max scaled, loss function: ",constants.loss_function )
print("ephochs: ",constants.number_of_epochs)
print("start : ", constants.start_date)
print("file: norm-vs-ann2-ti.py")
print("layers: 2 sigmoid")
print("rollingwindow:" , constants.rollingWindow)

print("--------------------------------------------------------------------")

print("(N-MC) totalTestSize :" ,totalTestSize)
print("(N-MC) totalTrainSize :" ,totalTrainSize)


print("(ANN) X_train_scaled :" ,X_train_scaled.shape)
print("(ANN) y_train_scaled :" ,y_train_scaled.shape)

print("(ANN) X_test_scaled :" ,X_test_scaled.shape)
print("(ANN) y_test_unscaled :" ,y_test_unscaled.shape)
