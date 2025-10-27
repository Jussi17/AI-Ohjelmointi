# M20 Lue artikkeli ”How to Develop a CNN From Scratch for
# CIFAR-10 Photo Classification” ja täydennä sen avulla malli
# koodipohjaan. Läpimeno vaaditaan vähintään 65%. Huomioi,
# että kouluttaminen saattaa viedä aikaa muutaman tunninkin.
# Testaa koodisi siis ensin korvaamalla epochs=60 esimerkiksi
# arvolla epochs=2. Palauta sitten alkuperäinen arvo.
# HUOM: Videossa sanotaan, että tarkastusohjelma lataa
# my code.py:n tallentaman my code.h5-tiedoston. Näin ei
# kuitenkaan enää ole, vaan tarkastus toimii kuten muidenkin
# tehtävien kanssa.

# test harness for evaluating models on the cifar10 dataset
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD # type: ignore
import os


# plot diagnostic learning curves
def plot_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.legend()
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()  

def init_model():
    global model
    # load data
    basedir = os.path.dirname(__file__)
    X = np.load(os.path.join(basedir, 'trainx.npy'))
    Y = np.load(os.path.join(basedir, 'trainy.npy'))
    
    N_test = 2000
    testX = X[:N_test, :, :]  
    testY = Y[:N_test, :]  
    
    trainX = X[N_test:, :, :]  
    trainY = Y[N_test:, :]  
    
    del X, Y
      
    model = Sequential()
    
    # VGG3-style CNN
    
    # 1. Conv block
    model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # 2. Conv block
    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    # 3. Conv block
    model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    # Flatten + Dense
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    # Optimizer
    opt = SGD(learning_rate=0.001, momentum=0.9)  
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(trainX, trainY, epochs=60, batch_size=64, validation_data=(testX, testY), verbose=1)

    # Save model
    model.save('my_code.keras')

    return testX, testY, history


def load_my_model(filename='my_code.keras'):
    global model
    model=load_model(filename)

def compute_values(x): #x is vector of images
    global model
    return model.predict(x)

if __name__ == "__main__":
    x_test, y_test, history=init_model()

    #_, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred=compute_values(x_test)
    y_pred_idx=np.argmax(y_pred, axis=1)
    y_test_idx=np.argmax(y_test, axis=1)
    
    N_correct=(y_pred_idx==y_test_idx).sum()
    N_all=np.shape(y_test)[0]
    acc=N_correct/N_all
    print('Accuracy: %.3f'%(acc))

    #Uncomment for test purposes, comment out in final version!
    #plot_diagnostics(history)

