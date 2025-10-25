# M14 Muokkaa ohjelmaa, joka opettaa KNN-luokittimen tunnistamaan
# käsinkirjoitettuja numeroita (MNIST-aineistosta). Tehtävänäsi
# on pienentää aineiston dimensio 32:een. Tavoitteena on saada
# vähintään 95% pakatun testiaineiston merkeistä tunnistettua.
# Älä muokkaa ohjelmaa muualta kuin rivien 15. . . 25 välistä.

import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

reduced_N=32

#Set value range -1..1
train_X = train_X.astype('float32') / 127.5 - 1
test_X = test_X.astype('float32') / 127.5 - 1

#Convert figures to vectors
train_X = train_X.reshape((train_X.shape[0], -1))
test_X = test_X.reshape((test_X.shape[0], -1))

#Compute reduced PCA

pca = PCA(n_components=reduced_N)
train_X_packed = pca.fit_transform(train_X)
test_X_packed = pca.transform(test_X)

#End of your code
########################################################
#Do not modify lines below this point!




#Save packed data
print('Save packed data')
np.save('packed_train.npy', train_X_packed)
np.save('packed_test.npy', test_X_packed)

if len(sys.argv)==1:
    #Test quality
    print('Train model')
    model = KNeighborsClassifier(n_neighbors = 11)
    model.fit(train_X_packed, train_Y)

    print('Compute predictions')
    pred = model.predict(test_X_packed)
    acc = accuracy_score(test_Y, pred)

    print('Accuracy =',acc)
