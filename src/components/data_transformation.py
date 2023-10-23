import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformerobject(self):
        try:
            num_columnas = ['Arbolado Adulto', 
                            'Renuevo', 
                            'Arbustivo', 
                            'Herbáceo', 
                            'Hojarasca',
                            'Segundos duración', 
                            'Segundos detección', 
                            'Segundos llegada']
            cat_columnas = ['Región', 
                            'Causa', 
                            'Causa especifica', 
                            'Tipo de incendio',
                            'Tipo Vegetación', 
                            'Régimen de fuego']
        
            
            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Columnas categóricas: {cat_columnas}')
            logging.info(f'Columnas numéricas: {num_columnas}') 

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columnas),
                    ('cat_pipeline', cat_pipeline, cat_columnas)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Lectura de datos de entrenamiento y prueba exitosa')

            preprocessing_obj=self.get_data_transformerobject()

            logging.info('Obtención del objeto de preprocesamiento')

            target_column = 'Tipo impacto'
            
            input_feature_train_df = train_df.drop(columns=['Año',
                            'Latitud',
                            'Longitud',
                            'Clave Entidad',
                            'Clave Municipio',
                            'Clave Geográfica',
                            'Tamaño',
                            'Duración días',
                            'Fecha Inicio',
                            'Fecha Termino',
                            'Tipo impacto',
                            'Total hectáreas',
                            'Detección',
                            'Llegada',
                            'Duración',
                            'Días'],axis=1)
            target_feature_train_df = train_df[[target_column]]

            input_feature_test_df = test_df.drop(columns=['Año',
                            'Latitud',
                            'Longitud',
                            'Clave Entidad',
                            'Clave Municipio',
                            'Clave Geográfica',
                            'Tamaño',
                            'Duración días',
                            'Fecha Inicio',
                            'Fecha Termino',
                            'Tipo impacto',
                            'Total hectáreas',
                            'Detección',
                            'Llegada',
                            'Duración',
                            'Días'],axis=1)
            target_feature_test_df = test_df[[target_column]]

            logging.info(
                f'Aplicación del objeto de preprocesamiento en el conjunto de entrenamiento y prueba'
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            ordinal_encoder = OrdinalEncoder(categories=[['Impacto Mínimo','Impacto Moderado', 'Impacto Severo']])
            target_feature_train_arr = ordinal_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = ordinal_encoder.transform(target_feature_test_df)

            target_feature_train_arr = target_feature_train_arr.reshape(-1)
            smote = SMOTE()
            input_feature_train_arr, target_feature_train_arr = smote.fit_resample(input_feature_train_arr, target_feature_train_arr)

            logging.info(f'Objeto preprocesado guardado')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                input_feature_train_arr, # X_train
                target_feature_train_arr, # y_train
                input_feature_test_arr, # X_test
                target_feature_test_arr, # y_test
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)