"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande
Description: 
Concepts Used:

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext


rdd = sc.textFile('/content/drive/My Drive/Project/ap_by_poll/42401/*')

def split_data(x):
  data = x.split(',')
  return data[0], (data[1:],)

def sorted_data(data):
  county = data[0]
  sort_data = sorted(data[1], key=lambda x: x[0])
  X = []
  for s in sort_data:
    X.append(float(s[1]))

  return county, X

group_by_county = rdd.map(split_data).reduceByKey(lambda x,y : x+y).map(sorted_data)

one_county = group_by_county.filter(lambda x: x[0] == '06037').take(1)[0]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def forecast(data):
  county = data[0]
  dataset = data[1]
  epochs= 50

  # reframed = series_to_supervised(dataset, 8, 8)
  # values = reframed.values
  X = []
  y = []
  for i in range(len(dataset)-14):
    X.append([dataset[j] for j in range(i, i+14)])
    # y.append([dataset[j] for j in range(i+10,i+15)])

  train = np.array(X[:-4000])
  test = np.array(X[-4000:])

#   n_obs = 8 * 13
  train_X, train_y = train[:, :-7], train[:, -7:]
  test_X, test_y = test[:, :-7], test[:, -7:]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  model = Sequential()
  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dense(7))
  model.compile(loss='mae', optimizer='adam')
  # fit network
  history = model.fit(train_X, train_y, epochs=epochs, batch_size=24, validation_data=(test_X, test_y), verbose=2, shuffle=False)
  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()

  yhat = model.predict(test_X)

  avg_preds = {} 
  for idx, pred8 in enumerate(yhat):
      for ind, pred in enumerate(pred8):
          val = avg_preds.get(ind + idx, (0,0))
          avg_preds[ind+idx] = (val[0]+ pred, val[1]+1)


  pred = [v[0]/v[1] for k, v in avg_preds.items()]

  return county, pred

def forecast(data):
  county = data[0]
  dataset = data[1]
  epochs= 10

  # reframed = series_to_supervised(dataset, 8, 8)
  # values = reframed.values
  X = []
  y = []
  for i in range(len(dataset)-15):
    X.append([dataset[j] for j in range(i, i+15)])
    # y.append([dataset[j] for j in range(i+10,i+15)])

  train = np.array(X[:-3000])
  test = np.array(X[-3000:])

#   n_obs = 8 * 13
  train_X, train_y = train[:, :-5], train[:, -5:]
  test_X, test_y = test[:, :-5], test[:, -5:]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  model = Sequential()
  model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dense(5))
  model.compile(loss='mse', optimizer='adam')
  # fit network
  history = model.fit(train_X, train_y, epochs=epochs, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
  # plot history
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()

  yhat = model.predict(test_X)

  avg_preds = {} 
  for idx, pred8 in enumerate(yhat):
      for ind, pred in enumerate(pred8):
          val = avg_preds.get(ind + idx, 0)
          avg_preds[ind+idx] = pred


  pred = [v for k, v in avg_preds.items()]

  return county, pred

pred = forecast(one_county)

plt.subplots(figsize=(20,8))
plt.plot(one_county[1][-3000:])
plt.plot(pred[1])
plt.show()

one_county

# ARIMA example
from statsmodels.tsa.arima_model import ARIMA
from random import random
# contrived dataset
data = one_county[1]
# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data)+10, typ='levels')
print(yhat)

plt.subplots(figsize=(20,15))
# plt.plot(one_county[1])
plt.plot(yhat)
plt.show()