

import numpy as np 
import os
import tensorflow as tf 
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical




#between 101 and 60,000. Note that It will still take ages to run even with few samples, and that the low accuracy is going to make it throw a ton of exceptions which would normally break it. They're being caught because it was common in the first couple epochs, until the model made at least one prediction for each class. But with few samples included, it never reaches that point.
NUMBER_OF_SAMPLES_TO_USE = 500

cifar100 = tf.keras.datasets.cifar100  
(x_train, y_train), (x_val, y_val) = cifar100.load_data()
x_all = np.concatenate([x_train, x_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)
x_all = x_all[:NUMBER_OF_SAMPLES_TO_USE]
y_all = y_all[:NUMBER_OF_SAMPLES_TO_USE]
#Change data set sizes, save current train/val
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.75, random_state=12)
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)
np.save('xtrain75-25.npy',x_train)
np.save('ytrain75-25.npy',y_train)
np.save('xtest75-25.npy',x_test)
np.save('ytest75-25.npy',y_test)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.90, random_state=12)
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)
np.save('xtrain90-10.npy',x_train)
np.save('ytrain90-10',y_train)
np.save('xtest90-10.npy',x_test)
np.save('ytest90-10.npy',y_test)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.50, random_state=12)
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)
np.save('xtrainhalf12.npy',x_train)
np.save('ytrainhalf12.npy',y_train)
np.save('xvalhalf12',x_test)
np.save('yvalhalf12',y_test)

