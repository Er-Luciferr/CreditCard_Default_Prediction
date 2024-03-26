import sys, os
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.components import data_ingestion
from src.exception import CustomException
from src.logger import logging
from src.utils.utils import MainUtils
from src.constant import *
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder)
    transformed_train_file_path = os.path.join(artifact_dir, 'train.nyp')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.nyp')
    transformed_object_file_path = os.path.join(artifact_dir, 'preprocessor.pkl')

class DataTransformation:
    def __init__(self, feature_store_file_path):

        self.raw_data_dir = raw_data_dir
        self.DataTransformationConfig = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data_transformer_object(self):
        try:
            # define the steps for the preprocessor pipeline
            preprocessor = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='constant',fill_value=0)),
                    ('scaler',RobustScaler())
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self):
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.0"""

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )

        try:
            dataframe = pd.read_csv(self.raw_data_dir)

            X = dataframe.drop(labels=[TARGET_COLUMN], axis=1)
            y = dataframe[TARGET_COLUMN]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.30,random_state=0)

            preprocessor = self.get_data_transformer_object()

            scaled_X_train = preprocessor.fit_transform(X_train)
            scaled_X_test = preprocessor.transform(X_test)

            preprocessor_path = self.DataTransformationConfig.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path,exist_ok=True))
            
            self.utils.save_object(file_path=preprocessor_path,obj=preprocessor)
            
            train_arr = np.c_[scaled_X_train,np.array(X_train)]
            test_arr = np.c_[scaled_X_test,np.array(X_test)]
            
            return train_arr,test_arr,preprocessor_path


        except Exception as e:
            raise CustomException(e,sys)


