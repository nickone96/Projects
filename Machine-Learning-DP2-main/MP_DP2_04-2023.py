import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, HTML
from sklearn.decomposition import PCA
from ctgan import CTGAN

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# Function to train and evaluate the model
def train_and_evaluate_model(model_type, C_values, X_train, y_train, X_test, y_test, n_jobs = -1):
    if model_type == 'svm':
        clf = svm.LinearSVC(C=C_values[0], max_iter=10000)
    elif model_type == 'logistic regression':
        clf = LogisticRegression(C=C_values[0], solver= 'saga',  n_jobs=n_jobs)
    elif model_type == 'nn_relu':
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', alpha=C_values[0], max_iter=1000)
        nn.fit(X_train, y_train.ravel())
        clf = nn
    elif model_type == 'nn_tanh':
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='tanh', alpha=C_values[0], max_iter=1000)
        nn.fit(X_train, y_train.ravel())
        clf = nn
    elif model_type == 'nn_logistic':
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', alpha=C_values[0], max_iter=1000)
        nn.fit(X_train, y_train.ravel())
        clf = nn      
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    clf.fit(X_train, y_train.ravel())
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    return -test_acc 

    

# Function to run neural network experiment
def run_nn_experiment(C_values, activation, X_train, y_train, X_test, y_test):
    # convert C_values to a list if it's not already
    if not isinstance(C_values, list):
        C_values = [C_values]
        
    best_score = -1
    ## Finds best C value
    for C in C_values:
        nn = MLPClassifier(hidden_layer_sizes=(10,10), alpha=C, activation=activation, solver='adam', max_iter=1000, random_state=42)
        nn.fit(X_train, y_train.ravel())
        score = nn.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_nn = nn
    
    #print(f'Best C for neural network with {activation} activation: {best_nn.alpha:.3g}, best test accuracy: {best_score:.3f}\n')
    return -best_score 

def run_cnn(activation, X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0], X_test.shape[1],1)
    cnn = models.Sequential()
    cnn.add(layers.Conv1D(32, 3, activation=activation, input_shape=X_train[0].shape))
    cnn.add(layers.MaxPooling1D(2))
    cnn.add(layers.Conv1D(64, 3, activation=activation))
    cnn.add(layers.MaxPooling1D(2))
    cnn.add(layers.Conv1D(64, 3, activation=activation))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation=activation))
    cnn.add(layers.Dense(10))
    cnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    test_loss, test_acc = cnn.evaluate(X_test,  y_test, verbose=0)
    print(test_acc)

# Function to optimize C for a given model
def optimize_C_for_model(model_type, X_train, y_train, X_test, y_test):
    C_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    C_values_with_extra = C_values + [1000]
    best_C = None

    if model_type == 'svm':
        C_space = Real(1e-7, 100, prior='log-uniform')
        objective_fn = partial(train_and_evaluate_model, 'svm', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'logistic regression':
        C_space = Real(1e-7, 100, prior='log-uniform')
        objective_fn = partial(train_and_evaluate_model, 'logistic regression', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'nn_relu':
        C_space = Real(1e-7, 1000, prior='log-uniform')
        objective_fn = partial(run_nn_experiment, activation='relu', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'nn_tanh':
        C_space = Real(1e-7, 1000, prior='log-uniform')
        objective_fn = partial(run_nn_experiment, activation='tanh', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'nn_logistic':
        C_space = Real(1e-7, 1000, prior='log-uniform')
        objective_fn = partial(run_nn_experiment, activation='logistic', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
      
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    try:
        result = gp_minimize(objective_fn, [C_space], n_calls=20, random_state=42)
        best_C = result.x[0]
        best_score = -result.fun

    except:
        print(f'Error occurred during optimization for {model_type}')
    
    if best_C is not None:
        test_score = train_and_evaluate_model(model_type, [best_C], X_train, y_train, X_test, y_test)
    
    return best_C

# Function for AdaBoost Classifier
def adaboost(X_train, y_train, X_test, y_test, n_estimators=50,  learning_rate=1.0):
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    return clf, train_accuracy, test_accuracy

# Function for Random Forest Classifier
def random_forest(X_train, y_train, X_test, y_test, n_jobs = -1, n_estimators=500, max_depth=None, min_samples_split=2, random_state=None):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap = True, min_samples_split=min_samples_split, random_state=random_state, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    return clf, train_accuracy, test_accuracy

# Function to generate synthetic data using GANs
def generate_synthetic_data_gan(df, epochs = 200, threshold = 0.5):
    synthesizer = CTGAN(epochs = epochs)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(len(df))
    target_col = 'Target'
    
    synthetic_target_values = synthetic_data[target_col].values
    binary_target_values = (synthetic_target_values > threshold).astype(int)
    synthetic_data[target_col] = binary_target_values
    
    return synthetic_data

# Function to plot histograms
def plot_histograms(original_data, synthetic_data, feature):
    fig, ax = plt.subplots(figsize=(12, 4))  
    ax.hist(original_data[feature], bins=20, color='blue', alpha=0.7, label='Original Data')
    ax.hist(synthetic_data[feature], bins=20, color='orange', alpha=0.7, label='Synthetic Data')    
    ax.set_title(f'Original vs. Synthetic Data: {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.show()
    
    

# Main function to preprocess the dataset, optimize hyperparameters, train, and evaluate various models
def main():
    
    ## Dataset with ~9,000 training instances and an additional ~9,000 from GANs. Total training instances are 19418
    df = pd.read_csv('clean_data_2.csv')
    columns_to_encode = ['Occupation_type', 'Housing_type', 'Family_status', 'Education_type', 'Income_type', 'Gender']
    data_one_hot = pd.get_dummies(df[columns_to_encode], drop_first = True)
    df = pd.concat([df.drop(columns_to_encode, axis=1), data_one_hot], axis=1)
   
    ## Changes Datatypes to 'float32' to properly run GANs
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].astype('float32')
    
    
    
    ## REMOVE COMMENTS TO APPLY GANs (CREATES ADDITIONAL TRAINING INSTANCES)
    '''
    synthetic_df = generate_synthetic_data_gan(df) 
    y_synthetic = synthetic_df['Target'] 
    synthetic_df = synthetic_df.drop(['Target', 'ID'], axis=1) 
    X_synthetic = synthetic_df
    '''
    
    
    ## Assigning the target to y and dropping from the dataframe
    y = df['Target'] 
    df = df.drop(['Target', 'ID'], axis=1)     
    X = df
  
    
    ## REMOVE COMMENTS TO APPLY GANs (CREATES PLOTS SHOWING A COMPARISON BETWEEN ORIGINAL DATA AND SYNTHETIC DATA)
    '''
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for feature in numeric_features:
        plot_histograms(df, synthetic_df, feature)
        
    '''
    ## REMOVE COMMENTS TO APPLY GANs (CONCATs ORIGNAL DATA AND SNYTHETIC DATA)
    '''
    X = pd.concat([X, X_synthetic])
    y = pd.concat([y, y_synthetic])
    '''
    

    
    '''This block of code is used to see the importance of each feature'''
    #rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    #rf.fit(X,y)
    #importances = rf.feature_importances_
    #feature_names = X.columns    
    #for feature, importance in zip(feature_names, importances):
        #print(feature, ':', importance * 100 , "%")
    

    X = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler().fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    
    pca = PCA()
    pca.fit(X_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    
    '''Using PCA to transform the Data'''
    #pca = PCA(n_components=0.95)  # Retain 95% of the variance in the dataset
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)
    
 
    #Support Vector Machines
    svm_C = optimize_C_for_model('svm', X_train, y_train, X_test, y_test)
    print('Best C for SVM:', svm_C)
    svm_score = train_and_evaluate_model('svm', [svm_C], X_train, y_train, X_test, y_test)
    print('SVM Score', round(svm_score * -1, 6),'\n')
    
    # Logistic Regression  
    lr_C = optimize_C_for_model('logistic regression', X_train, y_train, X_test, y_test)
    print('Best C for Logistic Regression:' , lr_C) 
    lg_score= train_and_evaluate_model('logistic regression', [lr_C], X_train, y_train, X_test, y_test)
    print('lg_score', round(lg_score * -1,6),'\n')
    
    # Neural Network ReLU activation function  
    nn_relu_C = optimize_C_for_model('nn_relu', X_train, y_train, X_test, y_test)
    print('Best C for NN with ReLU:', nn_relu_C)
    nn_relu = run_nn_experiment(nn_relu_C, 'relu', X_train, y_train, X_test, y_test)
    print('nn relu score', round(nn_relu * -1, 6),'\n')
    
    # Neural Network tanh activation function 
    nn_tanh_C = optimize_C_for_model('nn_tanh', X_train, y_train, X_test, y_test)
    print('Best C for NN with tanh:', nn_tanh_C)
    nn_tanh = run_nn_experiment(nn_tanh_C, 'tanh', X_train, y_train, X_test, y_test)
    print('nn tanh score', round(nn_tanh * -1, 6),'\n')
    
    # Neural Network logistic activation function 
    nn_logistic_C = optimize_C_for_model('nn_logistic', X_train, y_train, X_test, y_test)
    print('Best C for NN with logistic:', nn_logistic_C)
    nn_log= run_nn_experiment(nn_logistic_C, 'logistic', X_train, y_train, X_test, y_test)
    print('nn log score', round(nn_log * -1, 6),'\n')
    
    # CNN ReLU activation function
    run_cnn("relu", X_train, y_train, X_test, y_test)
    
    # CNN tanh activation function
    run_cnn("tanh", X_train, y_train, X_test, y_test)
    
    # CNN logistic activation function
    run_cnn("sigmoid", X_train, y_train, X_test, y_test)
    
    #Adaboost Decision Tree base estimator
    clf, train_accuracy, test_accuracy = adaboost(X_train, y_train, X_test, y_test,  n_estimators=50, learning_rate=1.0)
    adaboost_score = test_accuracy
    print(f"AdaBoost n_estimators: {50} ")
    print(f"AdaBoost learning Rate: {1.0}")
    print(f"AdaBoost score: {adaboost_score:.6f}\n")
    
    clf, train_accuracy, test_accuracy = random_forest(X_train, y_train, X_test, y_test, n_estimators = 100)
    random_forest_score = test_accuracy
    print(f"Random Forest n_estimators: {500}")
    print(f"Random Forest score: {random_forest_score:.6f}")


    
    model_names = ['SVM', 'Log Regression', 'NN ReLU', 'NN Tanh', 'NN Logistic', 'AdaBoost', 'Random Forest']
    model_scores = [
    svm_score * -1,
    lg_score * -1,
    nn_relu * -1,
    nn_tanh * -1,
    nn_log * -1,
    adaboost_score ,
    random_forest_score
    ]   

    plt.plot(model_names, model_scores, marker='o')
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison')
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

if __name__ == '__main__':
    main()


