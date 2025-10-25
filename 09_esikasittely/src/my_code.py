# M9 Lue tiedosto weather data.csv ja poista siitä päiväys
# (utc timestamp-sarake). Poista tämän jälkeen rivit, joilla on
# puuttuvia tietoja. Tallenna tulos tiedostoon preprocessed.csv.
# (Data from Data Platform)

import sys
import time
import pandas
import os

basedir = os.path.dirname(__file__)
inputfile = os.path.join(basedir, 'weather_data.csv')
outputfile = os.path.join(basedir, 'preprocessed.csv')

data=pandas.read_csv(inputfile)

for col in data.columns:
    col_lower = col.lower()
    if 'utc' in col_lower and 'timestamp' in col_lower:
        data = data.drop(columns=[col])
        print(f"Poistettu sarake: {col}")
        break
data = data.dropna()

data.to_csv(outputfile)
