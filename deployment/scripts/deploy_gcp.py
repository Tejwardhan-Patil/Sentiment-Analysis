import os
import logging
from google.cloud import storage
from google.oauth2 import service_account
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Constants for GCP configurations
PROJECT_ID = 'project-id'
BUCKET_NAME = 'bucket-name'
MODEL_NAME = 'model-name'
REGION = 'region'
SERVICE_ACCOUNT_FILE = 'path-to-service-account.json'
RUNTIME_VERSION = '2.5'
PYTHON_VERSION = '3.7'
MODEL_FRAMEWORK = 'TENSORFLOW'

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load service account credentials
def load_service_account_credentials(service_account_file):
    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        logger.info("Service account credentials loaded successfully.")
        return credentials
    except FileNotFoundError:
        logger.error(f"Service account file {service_account_file} not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading service account: {str(e)}")
        raise

# Initialize GCP services (Storage and ML Engine)
def initialize_services(credentials):
    try:
        storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
        ml_engine = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)
        logger.info("GCP services initialized successfully.")
        return storage_client, ml_engine
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

# Upload model to GCP storage
def upload_model_to_gcs(storage_client, model_path, bucket_name, destination_blob_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(model_path)
        logger.info(f"Model {model_path} uploaded to {destination_blob_name} in {bucket_name}.")
    except Exception as e:
        logger.error(f"Error uploading model to GCS: {str(e)}")
        raise

# Check if the bucket exists
def check_bucket_exists(storage_client, bucket_name):
    try:
        storage_client.get_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} exists.")
        return True
    except Exception as e:
        logger.error(f"Bucket {bucket_name} does not exist: {str(e)}")
        return False

# Create the bucket if it doesn't exist
def create_bucket(storage_client, bucket_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        storage_client.create_bucket(bucket, location=REGION)
        logger.info(f"Bucket {bucket_name} created successfully in region {REGION}.")
    except Exception as e:
        logger.error(f"Error creating bucket: {str(e)}")
        raise

# Deploy the model to AI Platform
def deploy_model(ml_engine, model_name, model_uri, region):
    try:
        model_body = {
            "name": model_name,
            "regions": [region],
            "onlinePredictionLogging": True
        }
        request = ml_engine.projects().models().create(
            parent=f'projects/{PROJECT_ID}',
            body=model_body
        )
        response = request.execute()
        logger.info(f"Model {model_name} deployed successfully with response: {response}")

        # Version deployment
        version_body = {
            "name": model_name,
            "deploymentUri": model_uri,
            "runtimeVersion": RUNTIME_VERSION,
            "framework": MODEL_FRAMEWORK,
            "pythonVersion": PYTHON_VERSION
        }
        version_request = ml_engine.projects().models().versions().create(
            parent=f'projects/{PROJECT_ID}/models/{model_name}',
            body=version_body
        )
        version_response = version_request.execute()
        logger.info(f"Version of model {model_name} deployed successfully with response: {version_response}")
    except HttpError as http_err:
        logger.error(f"HTTP error during deployment: {http_err}")
        raise
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise

# List all models in the project
def list_models(ml_engine):
    try:
        models = ml_engine.projects().models().list(parent=f'projects/{PROJECT_ID}').execute()
        if 'models' in models:
            logger.info("Models in the project:")
            for model in models['models']:
                logger.info(f"Model: {model['name']}")
        else:
            logger.info("No models found in the project.")
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise

# Delete an existing model
def delete_model(ml_engine, model_name):
    try:
        request = ml_engine.projects().models().delete(name=f'projects/{PROJECT_ID}/models/{model_name}')
        response = request.execute()
        logger.info(f"Model {model_name} deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {str(e)}")
        raise

# Set environment variables
def set_environment_variables():
    os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SERVICE_ACCOUNT_FILE
    logger.info("Environment variables set.")

# Main deployment process
def main():
    try:
        # Set environment variables
        set_environment_variables()

        # Load credentials
        credentials = load_service_account_credentials(SERVICE_ACCOUNT_FILE)

        # Initialize services
        storage_client, ml_engine = initialize_services(credentials)

        # Check and create bucket if necessary
        if not check_bucket_exists(storage_client, BUCKET_NAME):
            create_bucket(storage_client, BUCKET_NAME)

        # Define model path and GCS destination
        model_path = 'path-to-local-model-file'
        destination_blob_name = 'path-in-gcs-bucket'
        model_uri = f'gs://{BUCKET_NAME}/{destination_blob_name}'

        # Upload model to GCS
        upload_model_to_gcs(storage_client, model_path, BUCKET_NAME, destination_blob_name)

        # Deploy the model on AI Platform
        deploy_model(ml_engine, MODEL_NAME, model_uri, REGION)

        # List all models in the project
        list_models(ml_engine)

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()