import time
start_time = time.time()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
from sklearn.model_selection import train_test_split
from trainResnet import Resnet, learning_rate_schedule, get_random_eraser
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
import csv
import tracemalloc
print(f"Imports take: {int((elapsed := time.time() - start_time)//3600):02}:{int((elapsed%3600)//60):02}:{int(elapsed%60):02}")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Your `PyDataset` class should call `super().__init__")

def build_cnn_model(num_class=100):
    # build the model
    model = tf.keras.models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))  # Input layer
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def split_attack_by_prediction(x,y):#Used to split a dataset by outcome
    split_attack_x = [[] for i in range(x[1].shape[0])]
    split_attack_y = [[] for i in range(x[1].shape[0])]
    for i in range(len(x)):
        maxposition = np.argmax(x[i])
        split_attack_x[maxposition].append(x[i])
        split_attack_y[maxposition].append(y[i])
    return split_attack_x, split_attack_y
    

def load_csv(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        keys = first_line.split(',')
        data = {key: [] for key in keys}

        for line in file:
            values = line.strip().split(',')
            values = [int(values[0])] + [float(x) for x in values[1:]]
            for key, value in zip(keys, values):
                data[key].append(value)
                
    return data
    
def write_accuracys(accuracies_tuple,first_half_of_file_name=""):#writes epoch and accuracy for attack types to file
    epoch,  xgboost_accuracy_together, knn_accuracy_together, cnn_accuracy_together, xgboost_accuracy_seperately, knn_accuracy_seperately,cnn_accuracy_seperately = accuracies_tuple
    filename = first_half_of_file_name + "_attack_accuracies.csv"
    if not os.path.isfile(filename):
        with open(filename,"w") as file:
            file.write("epoch,xgb1,knn1,cnn1,xgb2,knn2,cnn2\n0,.5,.5,.5,.5,.5,.5\n")
    with open(filename,"a") as file:
        file.write(f"{epoch},{xgboost_accuracy_together},{knn_accuracy_together},{cnn_accuracy_together},{xgboost_accuracy_seperately},{knn_accuracy_seperately},{cnn_accuracy_seperately}\n")

def write_history(history_obj,epoch,first_half_of_file_name=""):#Writes model history to file
    filename = first_half_of_file_name + "_history.csv"
    if not os.path.isfile(filename):
        with open(filename,"w") as file:
            file.write("epoch,"+",".join(history_obj.history.keys()) + "\n")
    with open(filename,"a") as file:
        #file.write(str(epoch)+","+",".join(str(value) for value in history_obj.history.values()) + "\n")
        file.write(str(epoch) + "," + ",".join(str(value[0]) if isinstance(value, list) else str(value) for value in history_obj.history.values()) + "\n")

        

def eval_seperately(model, train_x, train_y, val_x, val_y): #Split data set and evaluate all for each class, and average the accuracy
    train_x,train_y = split_attack_by_prediction(train_x,train_y)
    val_x, val_y = split_attack_by_prediction(val_x, val_y)
    train_acc = []
    val_acc = []
    for i in range(len(train_x)):
        if np.array(train_x[i]).size == 0 or np.array(train_y[i]).size == 0 or np.array(val_x[i]).size == 0 or np.array(val_y[i]).size == 0:
            pass
        else:
            try:
                model.fit(np.array(train_x[i]), np.array(train_y[i]).ravel())
                accuracy = model.score(np.array(val_x[i]), np.array(val_y[i]).ravel())
                #print(f'Validation Accuracy: {accuracy:.4f}')
                val_acc.append(accuracy)
            except Exception as e:
                print(f"Skipping this one for reason:{e}")
    return np.mean(val_acc)
def eval_together(model, train_x, train_y, val_x, val_y):#Eval with a single modle instead of one per class
    model.fit(np.array(train_x), np.array(train_y).ravel())
    accuracy = model.score(np.array(val_x), np.array(val_y).ravel())
    return accuracy
    
def test_model(model,attack_dataset):
    try:
        attack_train_x, attack_val_x, attack_train_y, attack_val_y = attack_dataset
        #xgboost
        attack_model = XGBClassifier(eval_metric='logloss', verbosity=1)
        xgboost_accuracy_seperately = eval_seperately(attack_model, attack_train_x, attack_train_y, attack_val_x, attack_val_y)
        xgboost_accuracy_together = eval_together(attack_model, attack_train_x, attack_train_y, attack_val_x, attack_val_y)
        #knn
        attack_model = KNeighborsClassifier(n_neighbors=10)
        knn_accuracy_seperately  = eval_seperately(attack_model, attack_train_x, attack_train_y, attack_val_x, attack_val_y)
        knn_accuracy_together = eval_together(attack_model, attack_train_x, attack_train_y, attack_val_x, attack_val_y)
        #mlp
        attack_model = MLPClassifier(hidden_layer_sizes=(200,100,50), max_iter=5,solver='adam',activation='relu',verbose=0, random_state=42)
        cnn_accuracy_seperately  = eval_seperately(attack_model, attack_train_x, attack_train_y, attack_val_x, attack_val_y)
        cnn_accuracy_together = eval_together(attack_model, attack_train_x, attack_train_y, attack_val_x, attack_val_y)
    except Exception as e:
        print(attack_train_x.shape, attack_val_x.shape, attack_train_y.shape, attack_val_y.shape)
        #Ran into memory issues at some point, this helped solve
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("[ Top memory-consuming lines ]")
        for stat in top_stats[:5]:
            print(stat)
        raise e
    return xgboost_accuracy_together, knn_accuracy_together, cnn_accuracy_together, xgboost_accuracy_seperately, knn_accuracy_seperately,cnn_accuracy_seperately






def test_model_and_save_results(target_model,total_epochs, test_per_x_epochs,datagen_enabled,dataset,current_name = "",starting_epoch = 0):
    x_train, y_train, x_test, y_test = dataset

    min_sample_amount = min(x_test.shape[0],x_train.shape[0])
    attack_y_true  = [[1] for _ in range(x_train.shape[0])]
    attack_y_false = [[0] for _ in range(x_test.shape[0])]
    datagen = ImageDataGenerator(width_shift_range=4,height_shift_range=4,horizontal_flip=True, preprocessing_function=get_random_eraser(p=1, pixel_level=True))
    datagen.fit(x_train)
    augmented_data = datagen.flow(x_train, y_train, batch_size=128)
    for i in range(starting_epoch,total_epochs):
        print(f"Beginning epoch {i} on model {current_name}")
        print(f"Time elapsed: {int((elapsed := time.time() - start_time)//3600):02}:{int((elapsed%3600)//60):02}:{int(elapsed%60):02}")
        if datagen_enabled:
            history = target_model.fit(augmented_data,validation_data=(x_test, y_test),epochs=1,verbose=1)
            target_model.save(current_name+"_model.keras")
            
        else:
            history = target_model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1, validation_data=(x_test, y_test))
            target_model.save(current_name+"_model.keras")
        
        #Update history
        write_history(history,i,current_name)

        #Update tests
        if i !=0 and i%test_per_x_epochs == 0:
            #Get models current predictions
            attack_x_true  = target_model.predict(x_train,batch_size = 256)
            attack_x_false = target_model.predict(x_test,batch_size = 256)
            #ensure same number of true and false
            attack_x_true = attack_x_true[:min_sample_amount]
            attack_x_false = attack_x_false[:min_sample_amount]
            attack_y_true = attack_y_true[:min_sample_amount]
            attack_y_false = attack_y_false[:min_sample_amount]
            
            #Combine true and false into one
            attack_x = np.concatenate([attack_x_true, attack_x_false], axis=0)
            attack_y = np.concatenate([attack_y_true, attack_y_false], axis=0)
            attack_x = np.array(attack_x)
            attack_y = np.array(attack_y)
            #split some for validation
            attack_dataset = train_test_split(attack_x,attack_y,test_size=1000,random_state=12)
            xgboost_accuracy_together, knn_accuracy_together, cnn_accuracy_together, xgboost_accuracy_seperately, knn_accuracy_seperately,cnn_accuracy_seperately =  test_model(target_model,attack_dataset)
            print(xgboost_accuracy_together, xgboost_accuracy_seperately)
            write_accuracys((i, xgboost_accuracy_together, knn_accuracy_together, cnn_accuracy_together, xgboost_accuracy_seperately, knn_accuracy_seperately,cnn_accuracy_seperately),current_name)
    return target_model


#for model in models
#	for datagen yes or no
#		for data in dataset(split)
#			test_model_and_save_results(model,datagen = False, dataset)



x_train = np.load('xtrainhalf12.npy')
y_train = np.load('ytrainhalf12.npy')
x_test = np.load('xvalhalf12.npy')
y_test = np.load('yvalhalf12.npy')
dataset50 = (x_train, y_train, x_test, y_test)

x_train = np.load('xtrain75-25.npy')
y_train = np.load('ytrain75-25.npy')
x_test = np.load('xtest75-25.npy')
y_test = np.load('ytest75-25.npy')
dataset75 = (x_train, y_train, x_test, y_test)

x_train = np.load('xtrain90-10.npy')
y_train = np.load('ytrain90-10.npy')
x_test = np.load('xtest90-10.npy')
y_test = np.load('ytest90-10.npy')
dataset90 = (x_train, y_train, x_test, y_test)


accuracies_file_extension = "_attack_accuracies.csv"
history_file_extension = "_history.csv"

cnn_model = build_cnn_model()
resnet_model = Resnet().get_model()
optimizer = SGD(learning_rate=.2, momentum=0.9)
resnet_model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

model_info_set = [
    [cnn_model,100,1,True,dataset50,"CNN_50percent_data_with_Datagen-100epoch"],
    [cnn_model,100,1,True,dataset75,"CNN_75percent_data_with_Datagen-100epoch"],
    [cnn_model,100,1,True,dataset90,"CNN_90percent_data_with_Datagen-100epoch"],
    [cnn_model,100,1,False,dataset50,"CNN_50percent_data_with_Datagen-100epoch"],
    [cnn_model,100,1,False,dataset75,"CNN_75percent_data_with_Datagen-100epoch"],
    [cnn_model,100,1,False,dataset90,"CNN_90percent_data_with_Datagen-100epoch"],
    [resnet_model,100,1,True,dataset50,"Resnet_50percent_data_with_Datagen-100epoch"],
    [resnet_model,100,1,True,dataset75,"Resnet_75percent_data_with_Datagen-100epoch"],
    [resnet_model,100,1,True,dataset90,"Resnet_90percent_data_with_Datagen-100epoch"],
    [resnet_model,100,1,False,dataset50,"Resnet_50percent_data_with_Datagen-100epoch"],
    [resnet_model,100,1,False,dataset75,"Resnet_75percent_data_with_Datagen-100epoch"],
    [resnet_model,100,1,False,dataset90,"Resnet_90percent_data_with_Datagen-100epoch"],
    ]

#Can't just loop this in the same program, alot of weights are saved in a hidden way and mess it up when looped, even if you resubstatiate the model each time. You have to run it once per instance.
test_model_and_save_results(*model_info_set[0])
















