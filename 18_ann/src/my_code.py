# M18 Täydennä koodipohja sellaiseksi, että compute value-funktio
# palauttaa opetetun funktion arvon. Älä muokkaa funktiota.
# Tavoitteena on saada MSE alle 0.01. Aja useampi testi varmistaaksesi,
# että arvo pysyy rajan alapuolella suurimmassa osassa testejä.


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

#Just for help...
def plot(x, y):
    plt.subplot(2,1,1)
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], marker='.')
    plt.title('y[:, 0]')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 1], marker='.')
    plt.title('y[:, 1]')
    plt.colorbar()

#Predict value(s) of the function
def compute_value(x):
    global model
    return model.predict(x)


def init_model():
    global model #Model variable
    basedir = os.path.dirname(__file__)
    x=np.load(os.path.join(basedir, 'x.npy'))
    y=np.load(os.path.join(basedir, 'y.npy'))

    #NOTE: All data is in x and y    
    
    #Split data into test and train sets. Use 15000 samples in train set.
    #Modify lines below
    x_train=x[:15000]
    x_test=x[15000:]
    y_train=y[:15000]
    y_test=y[15000:]

    #Create model (network). Insert more lines if required.        
    model=Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))
    
    #print model information
    model.summary()
    
    #Compile model, choose loss function. Model line below
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    #Teach model. Insert required parameters
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

    #plt.plot(history.history['loss'])
    #plt.semilogy()
    #plt.show()

    return x_test, y_test


if __name__ == "__main__":
    x_test, y_test=init_model()

    y_pred=compute_value(x_test)
    plot(x_test, y_pred)
    plt.show()

    print("MSE =", mean_squared_error(y_test ,y_pred))
