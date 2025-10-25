''' M10 
1 Lue tiedosto time series.csv (Data from Data Platform)
2 Hae datasta Saksan tuulivoiman todellista tuotantoa koskevat
tiedot. Näissä tiedoissa region=DE, variable=wind ja
attribute=generation actual.
3 Poista edellä käytetyt sarakkeet, koska nyt niiden arvo on kaikilla
riveillä sama.
4 Poista rivit, joilla on puuttuvia tietoja
5 Poista rivit, joilla on outliereita. Käytä z-testin kynnysarvona
4:ää. HUOM: Anna stats.zscore-funktiolle parametriksi vain
data-sarake. Ei kaikkia sarakkeita, kuten luentokalvojen
esimerkissä.
6 Normalisoi data-sarake siten, että sarakkeen pienin arvo on 0 ja
suurin 1. Voit käyttää tässä esimerkiksi sklearn.preprocessing
kirjastoa.
7 Jaa data opetus- ja testijoukkoon siten, että opetusjoukkoon
menee 70% käsitellystä aineistosta ja loput testijoukkoon. Talleta
joukot nimillä train.csv ja test.csv.'''

import pandas
from scipy import stats
import numpy as np
from sklearn import preprocessing
import os

basedir = os.path.dirname(__file__)
inputfile = os.path.join(basedir, 'time_series.csv')
trainfile = os.path.join(basedir, 'train.csv')
testfile = os.path.join(basedir, 'test.csv')

data=pandas.read_csv(inputfile)

data = data[(data['region'] == 'DE') & (data['variable'] == 'wind') & (data['attribute'] == 'generation_actual')]
data = data.drop(columns=['region', 'variable', 'attribute'])
data = data.dropna()

z_scores = np.abs(stats.zscore(data['data']))
data = data[z_scores < 4]

scaler = preprocessing.MinMaxScaler()
data[['data']] = scaler.fit_transform(data[['data']])

train_size = int(0.7 * len(data))
traindata = data.iloc[:train_size]
testdata = data.iloc[train_size:]

#Save train data
print("Save train data")
traindata.to_csv(trainfile) #Columns saved: row index, timestamp, data. Don't remove row index

#Save test data
print("Save test data")
testdata.to_csv(testfile) #Columns saved: row index, timestamp, data. Don't remove row index

