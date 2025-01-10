import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split
import pandas as pd

from car_prediction.entity.config_entity import DataIngestionConfig
from car_prediction.entity.artifact_entity import DataIngestionArtifact
from car_prediction.exception import carprediction
from car_prediction.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise carprediction(e, sys)

    def load_data_from_local(self) -> DataFrame:
        """
        Method Name :   load_data_from_local
        Description :   This method loads data from a local CSV file.
        
        Output      :   DataFrame containing the loaded data.
        On Failure  :   Write an exception log and then raise an exception.
        """
        try:
            logging.info(f"Loading data from local file: {self.data_ingestion_config.local_data_file_path}")
            
            if not os.path.exists(self.data_ingestion_config.local_data_file_path):
                raise FileNotFoundError(f"File not found at {self.data_ingestion_config.local_data_file_path}")
            
            dataframe = pd.read_csv(self.data_ingestion_config.local_data_file_path)
            logging.info(f"Loaded data with shape: {dataframe.shape}")
            return dataframe

        except Exception as e:
            raise carprediction(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio.
        
        Output      :   Train and test datasets are saved to specified file paths.
        On Failure  :   Write an exception log and then raise an exception.
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train-test split on the dataframe")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving train and test datasets to files.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Train and test datasets saved successfully.")
        
        except Exception as e:
            raise carprediction(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion process.
        
        Output      :   DataIngestionArtifact containing paths to train and test datasets.
        On Failure  :   Write an exception log and then raise an exception.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.load_data_from_local()
            logging.info("Loaded data from local file")

            self.split_data_as_train_test(dataframe)
            logging.info("Performed train-test split on the dataset")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise carprediction(e, sys)
