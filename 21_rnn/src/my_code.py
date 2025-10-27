# M21 savilahti.csv sisältää lämpötilan mittaustietoja noin kahden
# vuoden ajalta. Valmis funktio ennustaa lämpötilan annetun datan
# perusteella 1 tunnin päähän edellisestä mittauksesta. Tehtävänasi
# on muokata koodipohjaa siten, että RMSE on alle 2 testijoukolla.
# HUOM: Videossa sanotaan, että tarkastusohjelma lataa
# my code.py:n tallentaman my code.h5-tiedoston. Näin ei
# kuitenkaan enää ole, vaan tarkastus toimii kuten muidenkin
# tehtävien kanssa.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Code copied from https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/

from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import load_model
import os

# Prepare the input X and target Y
def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    return X, Y

def create_RNN(input_shape, trainX, trainY):
    model = Sequential()
    
    
    #Write model here
    #HINT: Temperature values are not in range -1 to 1, and therefore the last
    #      layer activation functions shall be neither relu or tanh.
    #      You can insert Dense layers after SimpleRNN-layer, and in addition,
    #      you can use, for example, relu activation functions in those layers.
    #      See rnn01.py.
    #See rnn01.py
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    

    #Modify training parameters if needed
    model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=2)
    
    model.save('my_code.keras')
    
    return model

def print_error(trainY, testY, train_predict, test_predict):    
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))    

# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('The Red Line Separates The Training And Test Examples')

def load_my_model(filename='my_code.keras'):
    global model
    model=load_model(filename)

def compute_values(X):
    return model.predict(X)
    
if __name__ == "__main__":
    df=read_csv('savilahti.csv')
    df.dropna(inplace=True)
    temp=df['Air temperature (degC)']
    temp=np.array(temp)
    
    #prediction in 3 days (72h)
    time_steps = 72
    split=150*time_steps
    train_data=temp[:split]
    test_data=temp[split:]
    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)
    
    # Create model and train
    model = create_RNN(input_shape=(time_steps,1), trainX=trainX, trainY=trainY)
    
    # make predictions
    train_predict = compute_values(trainX)
    test_predict = compute_values(testX)
    
    # Print error
    print_error(trainY, testY, train_predict, test_predict)
    
    #Plot result
    plot_result(trainY, testY, train_predict, test_predict)
