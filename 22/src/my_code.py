# M22 Täydennä koodipohjan malli, jolla ennustat sikiön terveydentilaa
# kolmiportaisella asteikolla (1-3). Asiantuntijoiden merkitsemät
# terveydentilat on talletettu aineistoon fetal health-sarakeeseen.
# Valitse itse sopiva malli ja kirjoita mallin käyttöön liittyvät
# koodit init model ja compute value funktioihin. Lisää tiedoston
# alkuun tarvittavien kirjastojen lataus. Älä muokkaa
# testiohjelman osuutta.
# Tavoitteena on saada testiohjelman käyttämällä aineistolla
# enintään 30 virheellistä luokkaa. Testiaineistossa on 132 tapausta.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Globals for model and scaling
model = None
x_scaler = None
x_train_min = None
drop_columns = ['column1', 'column2', 'etc']  # Muokkaa tarpeen mukaan

# Split data into X and Y
def splitXY(d):
    Y_column='fetal_health'
    return d.drop(Y_column, axis=1).to_numpy(), np.int32(d[Y_column]) 

# Load CSV and drop unused columns
def load_data(filename):
    data = pd.read_csv(filename)
    for col in drop_columns:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    return data    

# Initialize and train the model
def init_model(datafile):
    global model, x_scaler, x_train_min

    # Load dataframe
    data = load_data(datafile)
    x, y = splitXY(data)

    # Split into test/train
    N_test = 132
    x_test_raw = x[:N_test]
    y_test = y[:N_test] - 1  # <- nollapohjaiset luokat
    x_train_raw = x[N_test:]
    y_train = y[N_test:] - 1 

    # Scale features 0..1 based on train set
    x_train_min = x_train_raw.min(axis=0)
    x_scaler = x_train_raw.max(axis=0) - x_train_min
    x_train = (x_train_raw - x_train_min) / x_scaler
    x_test = (x_test_raw - x_train_min) / x_scaler

    # Build the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(x.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=0)

    return x_test, y_test

# Predict values for given x
def compute_value(x):
    global model, x_scaler, x_train_min
    x_scaled = (x - x_train_min) / x_scaler
    y_pred = model.predict(x_scaled, verbose=0)
    y_pred = np.argmax(y_pred, axis=1) + 1  
    return y_pred

# Validate with another file
def validate(filename):
    data_val = load_data(filename)
    x_val, y_val = splitXY(data_val)
    y_pred = compute_value(x_val)
    return y_pred, y_val

# Test program
if __name__ == "__main__":
    x_test, y_test = init_model('fetal_health.csv')

    y_pred = compute_value(x_test)
    errors = (y_pred != y_test).sum()
    print('Errors (test set):', errors, '/', len(y_test))

    y_pred, y_val = validate('validation.csv')
    errors = (y_pred != y_val).sum()
    print('Errors (validation set):', errors, '/', len(y_val))