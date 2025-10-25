# M16 Tiedosto grading.csv sisältää tiedot opiskelijoiden 3 eri harjoituksesta
# saamista pisteistä ja tiedon tentin läpäisemistä.
# • Kustakin harjoituksesta voi saada 0 . . . 6 pistettä.
# • Mikäli jonkin sarakkeen arvo on tyhjä, opiskelija ei ole
# osallistunut harjoitukseen tai tenttiin. Tällaista opiskelijaa ei tule
# huomioida aineistossa.
# Esikäsittele aineisto ja opeta sen perusteella SVM jakamaan opiskelijat
# kahteen luokkaan tentin läpäisyn perusteella. Jaa aineisto opetus- ja
# testiaineistoon haluamallasi tavalla.
# Tiedostossa assignments.csv on opiskelijoiden harjoituspisteet.
# Tehtäväsi on ennustaa vähintään 80% opiskelijoista oikea
# tenttitulos. Tallenna ennusteet tiedostoon prediction.csv, joka
# sisältääopiskelijan nimen 1. sarakkeessa ja ennusteen 2. sarakkeessa.

import pandas 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm

basedir = os.path.dirname(__file__)
filename1 = os.path.join(basedir, 'grading.csv')
train_fraction = 0.8
Y_column = 'Passed'

# Load data
data = pandas.read_csv(filename1)
print("Read data shape = " + str(data.shape))
print(data)

data.dropna(inplace=True)
print("After removing missing values: " + str(data.shape))
print(data)

X = data.drop(columns=['Name', Y_column])
y = data[Y_column]
print("Features shape = " + str(X.shape))
print("Target shape = " + str(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_fraction, random_state=42)

print("X_train shape = " + str(X_train.shape))
print("X_test shape = " + str(X_test.shape))
print("y_train shape = " + str(y_train.shape))
print("y_test shape = " + str(y_test.shape))

classifier = svm.SVC(kernel='rbf', C=10, gamma='scale')
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

#
#################################################

# Load real data
filename2 = os.path.join(basedir, 'assignments.csv')
data = pandas.read_csv(filename2)
names = data['Name']

# Remove name column
for col in ["Name"]:
    print("Remove " + col)
    data.drop(col, axis=1, inplace=True)
print()

print("Read data shape = " + str(data.shape))
print(data)
predY = classifier.predict(data)

# Create dataframe from numpy data
df = pandas.DataFrame({'Name': names, 'Passed': predY})
print(df)
df.to_csv(os.path.join(basedir, 'prediction.csv'), index=False)