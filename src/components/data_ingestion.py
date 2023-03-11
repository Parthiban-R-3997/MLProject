import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass ## Decorator  ## Decators can be used for defining a variable inside a class
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") ## str= is string type and the path where the files are getting stored and artifacts is a folder
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) ## os.path.dirname  is basically getting directory name with respect to specific path
            ##exist_ok =True is if the directory is already there we can keep the particular folder and we dont want to delete or create it again and again
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) ## saving raw data

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) ##train test split initiated

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) ##saving train_csv

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) ##saving test_csv

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)