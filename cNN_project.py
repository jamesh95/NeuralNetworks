import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from random import randrange
from random import seed
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import pandas as pd
import pickle

# Load the Optical recognition of handwritten digits dataset
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data[:, :]
y = digits.target

# Using train_test_split split the dataset 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Determine the number of input features and classes
n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))

#-----------------------------------------------------------------------------------------------------
# Neural Network model
def Neural_Network(X_train, y_train):
    # define the model
    nn = Sequential()
    nn.add(Dense(32, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    nn.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
    nn.add(Dense(10, activation='softmax'))
    # compile the model
    nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the model
    nn.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
    return nn
nn = Neural_Network(X_train, y_train)

# Evaluate the nn model
def get_nn_metrics():
    loss, acc = nn.evaluate(X_test, y_test, verbose=0)
    return(loss, acc)

# Reshape data for use in CNN
def cnn_Reshape(X_train, X_test):
    # Original dataset comes as 64 instead of 8x8
    cnn_X_train = X_train.reshape((X_train.shape[0], 8, 8))
    cnn_X_test = X_test.reshape((X_test.shape[0], 8, 8))
    cnn_X_train = cnn_X_train.reshape((cnn_X_train.shape[0], cnn_X_train.shape[1], cnn_X_train.shape[2], 1))
    cnn_X_test = cnn_X_test.reshape((cnn_X_test.shape[0], cnn_X_test.shape[1], cnn_X_test.shape[2], 1))
    cnn_in_shape = cnn_X_train.shape[1:]
    return cnn_X_train, cnn_X_test, cnn_in_shape
reshaped = cnn_Reshape(X_train, X_test)
cnn_X_train = reshaped[0]
cnn_X_test = reshaped[1]
cnn_in_shape = reshaped[2]

# Convolutional Neural Network Model
def Conv_Neural_Network(X_train, X_test, y_train, y_test):
    # define model
    cnn = Sequential()
    cnn.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=cnn_in_shape))
    cnn.add(MaxPool2D((2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(n_classes, activation='softmax'))
    # define loss and optimizer
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # fit the model
    cnn.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)
    return cnn
cnn = Conv_Neural_Network(cnn_X_train, cnn_X_test, y_train, y_test)

# evaluate the cnn model
def get_cnn_metrics():
    cnn_loss, cnn_acc = cnn.evaluate(cnn_X_test, y_test, verbose=0)
    return(cnn_loss, cnn_acc)

#-----------------------------------------------------------------------------------------------------
# Save a model
def save(model_name, filename):
    model_name.save(filename)

# Load a model
def load(model_name, filename):
    model_name = tf.keras.models.load_model(filename)

#-----------------------------------------------------------------------------------------------------
# Model Evaluation

# Cross Validation with 5 sub-samples, param either nn or cnn
def cross_val(model_type):
    X_ = X
    y_ = y
    X_split = []
    y_split = []
    fold_size = int(len(X)/5)
    #seed(1)
    misclassed = 0
    average = 0
    # Create subsamples
    for i in range(5):
        X_sub_sample = []
        y_sub_sample = []
        while len(X_sub_sample) < fold_size:
            num = randrange(len(X_))
            X_sub_sample.append(X_[num])
            y_sub_sample.append(y_[num])
            X_ = np.delete(X_, num, 0)
            y_ = np.delete(y_, num, 0)
        X_split.append(X_sub_sample)
        y_split.append(y_sub_sample)
    # iterate 5 times, assigning a different test set each time and returning results
    for i in range(5):
        X_split_copy = X_split
        y_split_copy = y_split
        X_test = np.array(X_split[i])
        y_test = np.array(y_split[i])
        X_split_copy = np.delete(X_split_copy, i, 0)
        y_split_copy = np.delete(y_split_copy, i, 0)
        new_X = [i for _list in X_split_copy for i in _list]
        new_y = [i for _list in y_split_copy for i in _list]
        new_X = np.array(new_X)
        new_y = np.array(new_y)
        if model_type == nn:
            nn_ = Neural_Network(new_X, new_y)
            loss, acc = nn_.evaluate(X_test, y_test, verbose=0)
            wrong = fold_size - int((fold_size/100) * (acc*100))
            misclassed = misclassed + wrong
            average = average + acc
        if model_type == cnn:
            reshaped = cnn_Reshape(new_X, X_test)
            new_X = reshaped[0]
            X_test = reshaped[1]
            cnn_in_shape = reshaped[2]
            cnn_ = Conv_Neural_Network(new_X, X_test, new_y, y_test)
            loss, acc = cnn_.evaluate(X_test, y_test, verbose=0)
            wrong = fold_size - int((fold_size/100) * (acc*100))
            misclassed = misclassed + wrong
            average = average + acc
    accuracy = average/5
    return accuracy, misclassed

nn_cv = cross_val(nn)
cnn_cv = cross_val(cnn)

# Confusion Matrix

# Get predictions on main test set
nn_predictions = nn.predict_classes(X_test)
cnn_predictions = cnn.predict_classes(cnn_X_test)

# Create pandas dataframe to display the confusion matrix
nn_data = {'Actual':y_test,'Predicted':nn_predictions}
cnn_data = {'Actual':y_test,'Predicted':cnn_predictions}

nn_df = pd.DataFrame(nn_data)
cnn_df = pd.DataFrame(cnn_data)

# Create confusion matrix for 2 new models and 2 models from assignment 1
nn_confusion_matrix = pd.crosstab(nn_df['Actual'], nn_df['Predicted'], rownames=['Actual'], colnames=['Predicted'])

cnn_confusion_matrix = pd.crosstab(cnn_df['Actual'], cnn_df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
cnn_confusion_matrix


# ROC Curves

# Get probabilities for the two new models
nn_probs = nn.predict(X_test)
cnn_probs = cnn.predict(cnn_X_test)

# Function that returns the ROC points (tpr, fpr) for the array of thresholds.
def ROC(preds):
    roc_points = []
    thresholds = [0, 0.01, 0.11, 0.24, 0.39, 0.51, 0.65, 0.76, 0.88, 0.95, 0.99, 1]
    for t in thresholds:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(preds)):
            if max(preds[i]) >= t and argmax(preds[i]) == 8:
                prediction = True
            else:
                prediction = False

            if prediction == True and y_test[i] == 8:
                tp = tp + 1
            if prediction == True and y_test[i] != 8:
                fp = fp + 1
            if prediction == False and y_test[i] != 8:
                tn = tn + 1
            if prediction == False and y_test[i] == 8:
                fn = fn + 1
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)
        roc_points.append([tpr, fpr])
    return roc_points

# ROC points for each of the models
nn_roc_points = ROC(nn_probs)
cnn_roc_points = ROC(cnn_probs)

# I've turned these points into a dataframe
nn_df = pd.DataFrame(nn_roc_points, columns=["TPR", "FPR"])
cnn_df = pd.DataFrame(cnn_roc_points, columns=["TPR", "FPR"])


# -------------------------------- Interface -----------------------------------
def interface():
    inputs=['f1', 'f3', 'f4', 'f5', 'train', 'q']
    print("---------------------------------------------------------------------------------------------------------")
    print("")
    print("The dataset used is the Optical recognition of handwritten digits dataset imported from sklearn.")
    print("")
    print("---------------------------------------------------------------------------------------------------------")
    print("")
    print("Enter '1' to load the Neural Network and see the accuracy metrics")
    print("Enter '2' to load the Convolutional Neural Network and see the accuracy metrics")
    print("Enter '3' to see cross validation results on 5 samples for each model")
    print("Enter '4' to see confusion matrices for each model")
    print("Enter '5' to see ROC curves for each model")
    print("Enter 'train' to train the models")
    print("Enter 'q' to quit the program")
    print("")
    print("---------------------------------------------------------------------------------------------------------")
    x = input()
    if x == '1':
        print("Neural Network Metrics")
        loss, acc = get_nn_metrics()
        print("")
        print('Accuracy: %.3f' % acc)
        print("")
        print("Enter any key to return to menu")
        c = input()
        interface()
    if x == '2':
        print("Convolutional Neural Network Metrics")
        cnn_loss, cnn_acc = get_cnn_metrics()
        print("")
        print('Accuracy: %.3f' % cnn_acc)
        print("")
        print("Enter any key to return to menu")
        c = input()
        interface()
    if x == '3':
        print("Cross Validation with 5 Sub-Samples")
        print("")
        nn_cv_info = nn_cv
        cnn_cv_info = cnn_cv
        print('Neural Network Accuracy: %.3f' % nn_cv_info[0])
        print('Neural Network Missclassified:', nn_cv_info[1])
        print("")
        print('Convolutional Neural Network Accuracy: %.3f' % cnn_cv_info[0])
        print('Convolutional Neural Network Missclassified:', cnn_cv_info[1])
        print("")
        print("Enter any key to return to menu")
        c = input()
        interface()
    if x == '4':
        print("Confusion Matrices for each model")
        print("")
        print('Neural Network Confusion Matrix:')
        print(nn_confusion_matrix)
        print("")
        print('Convolutional Neural Network Confusion Matrix:')
        print(cnn_confusion_matrix)
        print("")
        print("Enter any key to return to menu")
        c = input()
        interface()
    if x == '5':
        print("ROC Curve for 3 available models")
        plt.plot(nn_df.FPR, nn_df.TPR, color="red", label = "nn")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        plt.plot(cnn_df.FPR, cnn_df.TPR, color="green", label = "cnn")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        print("Enter any key to return to menu")
        c = input()
        interface()
    if x == 'train':
        print("")
        print("Which algorithm would you like to train? Please enter '1' for regular neural network or '2' for the one with convolutional layer.")
        train = input()
        if train == '1':
            new_nn = Neural_Network(X_train, y_train)
            print("Would you like to save this model? Enter 'yes' or 'no'.")
            sav = input()
            if sav == 'yes':
                print("Enter name for model - must be one word. ")
                name = input()
                save(new_nn, name)
                print("Enter any key to return to menu")
                c = input()
                interface()
            if sav == 'no':
                interface()
        if train == '2':
            new_cnn = Conv_Neural_Network(cnn_X_train, cnn_X_test, y_train, y_test)
            print("Would you like to save this model? Enter 'yes' or 'no'.")
            sav = input()
            if sav == 'yes':
                print("Enter name for model - must be one word. ")
                name = input()
                save(new_cnn, name)
                print("Enter any key to return to menu")
                c = input()
                interface()
            if sav == 'no':
                interface()
        elif train != '1' and train != '2':
            print("Invalid input")
            print("Enter any key to return to menu")
            c = input()
            interface()
    if x == 'q':
        sys.exit(0)
    if x not in inputs:
        print("Invalid input")
        print("Enter any key to return to menu")
        c = input()
        interface()
interface()
