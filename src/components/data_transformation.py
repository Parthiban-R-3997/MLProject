import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ##for missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), ##Handling missing values
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),##most_frequent is MODE.
                ("one_hot_encoder",OneHotEncoder()), ##converting categorical to numerical orlse we can use target guided encoder also. If less number of categories are involved then go with one hot encoder
                ("scaler",StandardScaler(with_mean=False)) ##Not neccessary for categorical part.
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path): ##train_path and test_path are coming from data_ingestion.py

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] ##np.c means numpy in column wise concatenation
            #np.c_[] is a numpy function that is used to concatenate arrays along the second axis (horizontally)
            logging.info(f"Saved preprocessing object.")
            
            ##save_object is a function call and is coming from utils.py
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path, ##path to save the pkl file
                obj=preprocessing_obj ##This is the object with the model that has all transformations.In short we are saving this pickle name in hard disk.

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

"""
The os.path.dirname() function and the os.makedirs() function are both used for creating directories, but they serve different purposes.

os.path.dirname() function returns the directory part of a file path. For example, if the file_path is /home/user/filename.txt, the os.path.dirname(file_path) call will return /home/user. Essentially, this function extracts the directory component from the given file path.

On the other hand, os.makedirs() function creates a directory and all its parent directories if they do not already exist. The exist_ok=True argument ensures that the function will not raise an error if the directory already exists.

In the code snippet you provided, dir_path = os.path.dirname(file_path) extracts the directory component from the given file_path and saves it to the dir_path variable. Then, os.makedirs(dir_path, exist_ok=True) creates the directory and its parent directories if they don't already exist.

Overall, os.path.dirname() and os.makedirs() are used together in this code to ensure that the directory where the pickle file will be saved exists before attempting to write the file to disk.
"""        