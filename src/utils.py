import os 
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)   
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, 
                                                  modelos):
    try:
        accuracy = {}
        recall = {}
        f1 = {}
        cf = {}

        for i in range(len(list(modelos))):
            modelo = list(modelos.values())[i]
            modelo.fit(X_train, y_train)

            y_train_pred = modelo.predict(X_train)
            y_test_pred = modelo.predict(X_test)

            accuracy_train = accuracy_score(y_train, y_train_pred)
            recall_train = recall_score(y_train, y_train_pred, average='macro')
            f1_train = f1_score(y_train, y_train_pred, average='macro')
            cf_matrix_train = confusion_matrix(y_train, y_train_pred)

            accuracy_test = accuracy_score(y_test, y_test_pred)
            recall_test = recall_score(y_test, y_test_pred, average='macro')
            f1_test = f1_score(y_test, y_test_pred, average='macro')
            cf_matrix_test = confusion_matrix(y_test, y_test_pred)

            accuracy[list(modelos.keys())[i]] = accuracy_test
            recall[list(modelos.keys())[i]] = recall_test
            f1[list(modelos.keys())[i]] = f1_test
            cf[list(modelos.keys())[i]] = cf_matrix_test

            return recall

    except Exception as e:
        raise CustomException(e, sys)

