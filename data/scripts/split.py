import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths to the raw and processed data directories
raw_data_path = os.path.join("data", "raw")
processed_data_path = os.path.join("data", "processed")

# Check if a directory exists, if not, create it
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        logger.info(f"Directory {directory} does not exist. Creating it.")
        os.makedirs(directory)
    else:
        logger.info(f"Directory {directory} already exists.")

# Load the raw data with error handling
def load_data(file_name):
    try:
        file_path = os.path.join(raw_data_path, file_name)
        logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f"File {file_name} not found in {raw_data_path}. Ensure the file exists.")
        raise e
    except pd.errors.EmptyDataError:
        logger.error(f"The file {file_name} is empty.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the file {file_name}: {str(e)}")
        raise e

# Split the data into training, validation, and test sets
def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    try:
        logger.info("Splitting the data into training, validation, and test sets.")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
        logger.info(f"Data split successful: {len(train_data)} train samples, {len(val_data)} validation samples, {len(test_data)} test samples.")
        return train_data, val_data, test_data
    except Exception as e:
        logger.error(f"An error occurred during data splitting: {str(e)}")
        raise e

# Save the split datasets
def save_split_data(train_data, val_data, test_data, prefix="dataset"):
    try:
        ensure_directory_exists(processed_data_path)
        train_file = os.path.join(processed_data_path, f"{prefix}_train.csv")
        val_file = os.path.join(processed_data_path, f"{prefix}_val.csv")
        test_file = os.path.join(processed_data_path, f"{prefix}_test.csv")

        logger.info(f"Saving training data to {train_file}")
        train_data.to_csv(train_file, index=False)

        logger.info(f"Saving validation data to {val_file}")
        val_data.to_csv(val_file, index=False)

        logger.info(f"Saving test data to {test_file}")
        test_data.to_csv(test_file, index=False)

        logger.info("Data saving complete.")
    except Exception as e:
        logger.error(f"An error occurred while saving the data: {str(e)}")
        raise e

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into training, validation, and test sets.")
    parser.add_argument('--file_name', type=str, required=True, help="The name of the file containing the raw data.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the data to include in the test set.")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of the training data to include in the validation set.")
    parser.add_argument('--prefix', type=str, default="dataset", help="Prefix for the output files.")
    parser.add_argument('--random_state', type=int, default=42, help="Random state for data splitting.")
    return parser.parse_args()

# Main function to execute the data splitting
def main():
    try:
        args = parse_args()

        # Log the input parameters
        logger.info(f"Starting data split with the following parameters: "
                    f"file_name={args.file_name}, test_size={args.test_size}, val_size={args.val_size}, "
                    f"prefix={args.prefix}, random_state={args.random_state}")

        data = load_data(args.file_name)
        if data is None or data.empty:
            logger.error("No data found. Exiting.")
            return

        train_data, val_data, test_data = split_data(data, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)

        save_split_data(train_data, val_data, test_data, prefix=args.prefix)
        logger.info("Data splitting process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise e

# Utility function to display dataset information
def display_dataset_info(df):
    logger.info(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
    logger.info(f"First few rows of the dataset: \n{df.head()}")

# Handling cases where data may have missing values
def handle_missing_values(df):
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        logger.warning(f"Data contains {total_missing} missing values. Filling missing values with placeholder.")
        df.fillna("MISSING", inplace=True)
    return df

# Checking data consistency after splitting
def check_data_consistency(train_data, val_data, test_data):
    logger.info("Checking data consistency across splits.")
    assert len(train_data) > 0, "Training data is empty."
    assert len(val_data) > 0, "Validation data is empty."
    assert len(test_data) > 0, "Test data is empty."
    logger.info("Data consistency check passed.")

# Script entry point
if __name__ == "__main__":
    main()