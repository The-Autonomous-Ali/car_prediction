import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from car_prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from car_prediction.entity.config_entity import DataTransformationConfig
from car_prediction.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from car_prediction.exception import carprediction
from car_prediction.logger import logging
from car_prediction.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from car_prediction.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise carprediction(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from local file: {file_path}")
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise carprediction(f"File not found at path: {file_path}", sys)
        except Exception as e:
            raise carprediction(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore')
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, and OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            return preprocessor

        except Exception as e:
            raise carprediction(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Check if 'name' column exists before creating 'brand'
                if 'name' in train_df.columns:
                    train_df['brand'] = train_df['name'].apply(lambda x: x.split()[0])
                    train_df.drop(columns=['name'], inplace=True)
                    logging.info("Added 'brand' column and dropped 'name' column from Training dataset")

                if 'name' in test_df.columns:
                    test_df['brand'] = test_df['name'].apply(lambda x: x.split()[0])
                    test_df.drop(columns=['name'], inplace=True)
                    logging.info("Added 'brand' column and dropped 'name' column from Test dataset")

                # Drop specified columns
                drop_cols = self._schema_config['drop_columns']
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN] + drop_cols, errors='ignore', axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN] + drop_cols, errors='ignore', axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Dropped specified columns from Training and Test datasets")

                # Transform target column
                logging.info("Transforming target column using TargetValueMapping")

                target_value_mapping_train = TargetValueMapping(target_column=TARGET_COLUMN, dataframe=train_df)
                train_df = target_value_mapping_train.apply_mapping()

                target_value_mapping_test = TargetValueMapping(target_column=TARGET_COLUMN, dataframe=test_df)
                test_df = target_value_mapping_test.apply_mapping()

                target_feature_train_df = train_df[f"{TARGET_COLUMN}_numeric"]
                target_feature_test_df = test_df[f"{TARGET_COLUMN}_numeric"]

                logging.info("Successfully transformed target column in training and testing datasets")

                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object and transformed datasets")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise carprediction(e, sys) from e
