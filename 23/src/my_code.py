# M23 Täydennä koodipohjan malli, jolla ennustat sydänpotilaiden
# selviytymistä seurantajakson aikana (sarake DEATH EVENT).
# Valitse itse sopiva malli ja kirjoita mallin käyttöön liittyvät
# koodit init model ja compute value funktioihin. Lisää tiedoston
# alkuun tarvittavien kirjastojen lataus. Älä muokkaa
# testiohjelman osuutta.
# Tavoitteena on saada testiohjelman käyttämällä aineistolla
# enintään 10 virheellistä tulosta. Testiaineistossa on 34 tapausta.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Globals
model = None
scaler = None

# Split data into X and Y
def splitXY(d):
    Y_column = 'DEATH_EVENT'
    return d.drop(Y_column, axis=1).to_numpy(), np.int32(d[Y_column])

# Load CSV
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Initialize and train model
def init_model(datafile):
    global model, scaler

    data = load_data(datafile)
    x, y = splitXY(data)

    # Scale features
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # Stratified train/test split to keep class balance
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=34, stratify=y, random_state=42
    )

    # Build model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(x.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=0)

    return x_test, y_test

# Predict values for given x
def compute_value(x):
    global model, scaler
    x_scaled = scaler.transform(x)
    pred_prob = model.predict(x_scaled, verbose=0)
    pred_y = (pred_prob > 0.5).astype(int).flatten()
    return pred_y

# Validate with another file
def validate(filename):
    data_val = load_data(filename)
    x_val, y_val = splitXY(data_val)
    y_pred = compute_value(x_val)
    return y_pred, y_val

# Test program
if __name__ == "__main__":
    x_test, y_test = init_model('train.csv')
    y_pred = compute_value(x_test)
    errors = (y_pred != y_test).sum()
    print('Errors (test set):', errors, '/', len(y_test))

    y_pred, y_val = validate('validation.csv')
    errors = (y_pred != y_val).sum()
    print('Errors (validation set):', errors, '/', len(y_val))




    
    
