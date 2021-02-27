import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras


def generate_data(size, sigma=0.1):
    x = np.random.uniform(0, 2*np.pi, size)
    y = np.sin(x) + np.random.normal(scale=sigma, size=size)
    return x, y

for units in range(10,51,10):
    n_train = 100000
    n_test = 1000
    x_training, y_training = generate_data(n_train)
    mean, std = np.mean(x_training), np.std(x_training)
    x_training = (x_training - mean)/std


    ##################################################################

    network = keras.models.Sequential()
    ### **** Here, create and train the network using the keras api
    network.add(keras.layers.Dense(20,activation='relu' ,input_dim=1))
    network.add(keras.layers.Dense(10,activation='tanh'))
    network.add(keras.layers.Dense(1,activation='linear'))
    network.summary()
    network.compile(loss='mse',optimizer='sgd')
    history=network.fit(x_training,y_training,epochs=30,batch_size=200)
    ###################################################################

    x_test, y_test = generate_data(n_test)
    isort = np.argsort(x_test)
    plt.scatter(x_training * std + mean, y_training, label='Training distribution')
    plt.scatter(x_test, y_test, label='Test distribution')
    plt.plot(x_test[isort], np.sin(x_test[isort]), 'k--', label='True function')
    plt.plot(x_test[isort], network.predict((x_test[isort] - mean)/std), color='red', label='Network prediction')
    plt.legend(loc='best')
    plt.savefig('sine_test'+str(units/10)+'.png')
    plt.figure()
