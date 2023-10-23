import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

from xgboost import XGBClassifier

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            logging.info('Carga de datos de entrenamiento y testeo')

            modelos = {'Logistic Regression': LogisticRegression(max_iter=100000),
                       'Ridge': LogisticRegression(penalty='l2', solver='newton-cg', C=1.0, max_iter=100000),
                       'Decision Tree': DecisionTreeClassifier(),
                       'Random Forest': RandomForestClassifier(),
                       'XGBClassifier': XGBClassifier(),
                       'AdaBoostClassfier': AdaBoostClassifier()
                       }
            recall_dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                  modelos=modelos)
            
            best_model_recall = max(sorted(recall_dict.values())) 

            best_model_name = list(recall_dict.keys())[list(recall_dict.values()).index(best_model_recall)]

            best_model = modelos[best_model_name]

            if best_model_recall<0.6:
                raise CustomException('El mejor modelo no supera el 60% de recall', sys)
            
            logging.info(f'El mejor modelo es {best_model_name} con un recall de {best_model_recall}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            recall_sc = recall_score(y_test, predicted, average='macro')

            return recall_sc

        
        except Exception as e:
            raise CustomException(e, sys)