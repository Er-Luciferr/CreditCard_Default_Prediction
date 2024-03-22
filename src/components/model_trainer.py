import sys,os
from typing import Generator,Tuple,List
import pandas as pd
import numpy as np 

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split
from dataclasses import dataclass

from src.components import data_ingestion,data_transformation
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import MainUtils
from src.constant import *

@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join(artifact_folder,'model.pkl')
    expected_accuracy = 0.45
    model_config_file.path = os.path.join('config','model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
        self.utils = MainUtils()
        self.model = {}