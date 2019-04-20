import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import math
from keras.utils.vis_utils import plot_model
import uuid
from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from polyaxon_client.tracking.contrib.keras import PolyaxonKeras
import argparse

def clearY(y):
    clean_input = np.array([]).reshape(0, 1)
    for data in y:
        pos1 = data[0]
        pos2 = data[1]
        pos3 = data[2]
        if  pos1 == 1 and pos2 == 0 and pos3 ==0:
                clean_input = np.vstack((clean_input, [1]))
        else:
                clean_input = np.vstack((clean_input, [0]))
    return clean_input

def evaluate(true_y, pred_y):
    true_classes = []
    for array in true_y:
        if np.array_equal(array,[1, 0, 0]):
            true_classes.append(0)
        elif np.array_equal(array,[0, 1, 0]):
            true_classes.append(1)
        else:
            true_classes.append(2)
        
    CR, CA, PFA, GFA, FR, k = 0, 0, 0, 0, 0, 3.0
    for idx, prediction in enumerate(pred_y):
        # the students answer is correct in meaning and language
        # the system says the same -> accept
        if true_classes[idx] == 0 and prediction == 1:
            CA += 1
        # the system says correct meaning wrong language -> reject
        elif true_classes[idx] == 0 and prediction == 0:
            FR += 1
        # the system says incorrect meaning and incorrect language -> reject
        elif true_classes[idx] == 0 and prediction == 0:
            FR += 1

        # students answer is correct in meaning and wrong in language
        #The system says the same -> reject
        elif true_classes[idx] == 1 and prediction == 0:
            CR += 1
        # the system says correct meaning and correct language -> accept
        elif true_classes[idx] == 1 and prediction == 1:
            PFA += 1
        # the system says incorrect meaning and incorrect language -> reject
        elif true_classes[idx] == 1 and prediction == 0:
            CR += 1

        # students answer is incorrect in meaning and incorrect in language
        # the system says the same -> reject
        elif true_classes[idx] == 2 and prediction == 0:
            CR += 1
        # the system says correct meaning correct language -> accept
        elif true_classes[idx] == 2 and prediction == 1: 
            GFA += 1
        # the system says correct meaning incorrect language -> reject
        elif true_classes[idx] == 2 and prediction == 0:
            CR += 1

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA
    IncorrectRejectionRate = CR / ( CR + FA + 0.0 )
    CorrectRejectionRate = FR / ( FR + CA + 0.0 )
    # Further metrics
    Z = CA + CR + FA + FR
    Ca = CA / Z
    Cr = CR / Z
    Fa = FA / Z
    Fr = FR / Z
    
    P = Ca / (Ca + Fa)
    R = Ca / (Ca + Fr)
    SA = Ca + Cr
    F = (2 * P * R)/( P + R)
    
    RCa = Ca / (Fr + Ca)
    RFa = Fa / (Cr + Fa)
    
    D = IncorrectRejectionRate / CorrectRejectionRate
    Da = RCa / RFa
    Df = math.sqrt((Da*D))
    return Df

experiment = Experiment()

# 0. Read Args
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)

    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float)
    
    parser.add_argument(
        '--dropout',
        default=0.25,
        type=float)

    parser.add_argument(
        '--num_epochs',
        default=1,
        type=int)

args = parser.parse_args()
arguments = args.__dict__
batch_size = arguments.pop('batch_size')
learning_rate = arguments.pop('learning_rate')
dropout = arguments.pop('dropout')
num_epochs = arguments.pop('num_epochs')

# 1. Load Data
train_x = np.loadtxt('/data/shared-task/vec_train_x.csv' ,delimiter=',',usecols=range(11)[1:])
train_y = clearY(np.loadtxt('/data/shared-task/vec_train_y.csv', delimiter=',',usecols=range(4)[1:]))
dev_test_x = np.loadtxt('/data/shared-task/vec_test_x.csv', delimiter=',',usecols=range(11)[1:])
dev_test_y = np.loadtxt('/data/shared-task/vec_test_y.csv', delimiter=',',usecols=range(4)[1:])

experiment.log_data_ref(data=train_x, data_name='train_x')
experiment.log_data_ref(data=train_y, data_name='train_y')
experiment.log_data_ref(data=dev_test_x, data_name='dev_test_x')
experiment.log_data_ref(data=dev_test_y, data_name='dev_test_y')

# 2. Preporcessing
seed = 7
np.random.seed(seed)
sc = StandardScaler()
scaled_train_x = sc.fit_transform(train_x)
scaled_dev_test_x = sc.transform(dev_test_x)

# 3. Build the NN
classifier = Sequential()
classifier.add(Dense(64, activation='relu', input_dim=10))
classifier.add(Dropout(dropout))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(dropout))
classifier.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 4. Traing the Model
metrics = classifier.fit(scaled_train_x, train_y, batch_size = batch_size, epochs = num_epochs, validation_split=0.1, callbacks=[PolyaxonKeras(experiment=experiment)])

# 5. D-Evaluation
dev_y_pred = classifier.predict_classes(scaled_dev_test_x)
experiment.log_metrics(d_full=evaluate(dev_test_y, dev_y_pred))