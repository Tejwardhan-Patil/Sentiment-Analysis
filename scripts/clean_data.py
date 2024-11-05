import os
import shutil

# Define paths for the raw and processed directories
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
ANNOTATIONS_DIR = 'data/annotations'

# Define file types that need to be cleaned/organized
ALLOWED_FILE_TYPES = ['txt', 'csv', 'json']

def clean_raw_data():
    """
    Removes any files from the raw data directory that are not allowed.
    Organizes raw files by their file type.
    """
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = file.split('.')[-1].lower()
            
            # Remove files with disallowed extensions
            if file_extension not in ALLOWED_FILE_TYPES:
                os.remove(file_path)
            else:
                # Organize files by moving them into appropriate folders
                target_dir = os.path.join(PROCESSED_DATA_DIR, file_extension)
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(file_path, os.path.join(target_dir, file))

def ensure_annotations_exist():
    """
    Ensures that the annotations directory is properly organized.
    If annotations are missing or misnamed, raises an error.
    """
    if not os.path.exists(ANNOTATIONS_DIR):
        raise FileNotFoundError(f"{ANNOTATIONS_DIR} does not exist")
    
    annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.csv')]
    
    if len(annotation_files) == 0:
        raise ValueError("No annotation files found in annotations directory")

def main():
    # Clean and organize raw data
    clean_raw_data()
    
    # Ensure annotations are available
    ensure_annotations_exist()

if __name__ == '__main__':
    main()