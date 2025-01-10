import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from car_prediction.exception import carprediction
from car_prediction.logger import logging


class TargetValueMapping:
    def __init__(self, target_column: str, dataframe: DataFrame):
        """
        Initialize the TargetValueMapping class.
        :param target_column: The name of the target column in the dataset.
        :param dataframe: The dataset containing the target column.
        """
        self.target_column = target_column
        self.dataframe = dataframe
        self.mapping = self._create_mapping()
    
    def _create_mapping(self):
        """
        Create a mapping for all unique classes in the target column.
        :return: A dictionary mapping class labels to numeric values.
        """
        unique_classes = self.dataframe[self.target_column].unique()
        return {label: idx for idx, label in enumerate(unique_classes)}
    
    def apply_mapping(self):
        """
        Apply the mapping to the target column in the dataframe.
        :return: A new dataframe with the mapped target column.
        """
        self.dataframe[f"{self.target_column}_numeric"] = self.dataframe[self.target_column].map(self.mapping)
        return self.dataframe
    
    def filter_classes(self, max_class_value: int):
        """
        Filter the dataset to include only classes ≤ max_class_value.
        :param max_class_value: The maximum class value to retain in the dataset.
        :return: A filtered dataframe.
        """
        self.dataframe = self.dataframe[self.dataframe[f"{self.target_column}_numeric"] <= max_class_value]
        return self.dataframe
    
    def reverse_mapping(self):
        """
        Reverse the mapping for decoding numeric values back to original class labels.
        :return: A dictionary mapping numeric values back to class labels.
        """
        return dict(zip(self.mapping.values(), self.mapping.keys()))


class CarpredictionModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise carprediction(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
