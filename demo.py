import os
import sys

from carprice.components.data_ingestion import DataIngestion
from carprice.exception import CarException
from carprice.logger import logging
from carprice.config.configuration import Configuration

def main():
    try:
        # Load configuration
        config = Configuration()
        
        # Initialize data ingestion
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        
        # Start data ingestion process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Print the results
        print("Data Ingestion Artifact:")
        print(f"Train File Path: {data_ingestion_artifact.train_file_path}")
        print(f"Test File Path: {data_ingestion_artifact.test_file_path}")
        print(f"Is Ingested: {data_ingestion_artifact.is_ingested}")
        print(f"Message: {data_ingestion_artifact.message}")

    except CarException as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
