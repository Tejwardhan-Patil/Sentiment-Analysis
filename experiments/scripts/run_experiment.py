import os
import yaml
import argparse
import logging
import traceback
from models import train
from utils.metrics import calculate_metrics
from data.scripts.preprocess import preprocess_data
from data.scripts.split import split_data
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads the experiment configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}.")
        return config
    except FileNotFoundError as fnf_error:
        logger.error(f"Configuration file not found: {fnf_error}")
        raise
    except yaml.YAMLError as yaml_error:
        logger.error(f"Error in YAML parsing: {yaml_error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise

def validate_config(config):
    """Validates the loaded configuration for necessary keys."""
    required_keys = ['data', 'model', 'preprocessing', 'evaluation', 'output']
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required key in configuration: {key}")
            raise KeyError(f"Configuration must contain '{key}'")

def preprocess_and_split_data(config):
    """Preprocesses and splits data according to the configuration."""
    try:
        # Step 1: Preprocess Data
        logger.info("Starting data preprocessing...")
        data = preprocess_data(config['data']['raw_data_path'], config['preprocessing'])

        # Step 2: Split Data
        logger.info("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = split_data(data, config['data']['split'])

        logger.info("Data preprocessing and splitting completed successfully.")
        return train_data, val_data, test_data
    except Exception as e:
        logger.error(f"Error in preprocessing or splitting data: {e}")
        logger.debug(traceback.format_exc())
        raise

def train_model(train_data, val_data, config):
    """Trains the model as per the configuration."""
    try:
        model_name = config['model']['name']
        hyperparameters = config['model']['hyperparameters']

        logger.info(f"Training {model_name} model with hyperparameters: {hyperparameters}...")
        model = train.train_model(
            train_data=train_data,
            val_data=val_data,
            model_name=model_name,
            hyperparameters=hyperparameters
        )
        logger.info(f"Model {model_name} trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error in training model: {e}")
        logger.debug(traceback.format_exc())
        raise

def evaluate_model(model, test_data, config):
    """Evaluates the model using the test dataset and configuration."""
    try:
        evaluation_metrics = config['evaluation']
        logger.info("Evaluating model on test data...")
        predictions, metrics = calculate_metrics(model, test_data, evaluation_metrics)

        logger.info("Evaluation completed. Metrics calculated.")
        return predictions, metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        logger.debug(traceback.format_exc())
        raise

def save_results(predictions, metrics, config):
    """Saves predictions and evaluation metrics to the output directory."""
    try:
        output_dir = config['output']['results_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        predictions_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        metrics_path = os.path.join(output_dir, f'metrics_{timestamp}.yaml')

        with open(predictions_path, 'w') as pred_file:
            logger.info(f"Saving predictions to {predictions_path}...")
            for prediction in predictions:
                pred_file.write(f"{prediction}\n")

        with open(metrics_path, 'w') as metrics_file:
            logger.info(f"Saving metrics to {metrics_path}...")
            yaml.dump(metrics, metrics_file)

        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Error in saving results: {e}")
        logger.debug(traceback.format_exc())
        raise

def run_experiment(config):
    """Runs an experiment with the provided configuration."""
    try:
        # Step 1: Validate Configuration
        logger.info("Validating configuration...")
        validate_config(config)

        # Step 2: Preprocess and Split Data
        train_data, val_data, test_data = preprocess_and_split_data(config)

        # Step 3: Train Model
        model = train_model(train_data, val_data, config)

        # Step 4: Evaluate Model
        predictions, metrics = evaluate_model(model, test_data, config)

        # Step 5: Save Results
        save_results(predictions, metrics, config)
        
        logger.info("Experiment completed successfully.")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sentiment analysis experiment.')
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config file.')
    
    args = parser.parse_args()

    try:
        # Step 1: Load Configuration
        config = load_config(args.config)

        # Step 2: Run the Experiment
        run_experiment(config)

    except Exception as e:
        logger.error(f"Failed to run experiment: {e}")
        logger.debug(traceback.format_exc())