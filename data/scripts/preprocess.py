import os
import re
import string
import logging
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import json

logging.basicConfig(level=logging.INFO)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for input/output
raw_data_path = os.path.join('data', 'raw')
processed_data_path = os.path.join('data', 'processed')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    logging.info('Cleaning text')
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to tokenize text
def tokenize_text(text):
    logging.info('Tokenizing text')
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

# Function to stem tokens
def stem_tokens(tokens):
    logging.info('Stemming tokens')
    return [stemmer.stem(token) for token in tokens]

# Function to lemmatize tokens
def lemmatize_tokens(tokens):
    logging.info('Lemmatizing tokens')
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to normalize text
def normalize_text(text):
    logging.info('Normalizing text')
    # Convert to lowercase
    text = text.lower()
    # Remove accents and special characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to process a single text entry
def process_text(text, stem=False, lemmatize=False):
    logging.info('Processing a single text entry')
    text = clean_text(text)
    text = normalize_text(text)
    tokens = tokenize_text(text)
    
    if stem:
        tokens = stem_tokens(tokens)
    
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    return ' '.join(tokens), tokens

# Function to process a CSV file
def preprocess_csv(input_file, output_file, stem=False, lemmatize=False):
    logging.info(f'Preprocessing CSV file: {input_file}')
    # Load raw data
    data = pd.read_csv(os.path.join(raw_data_path, input_file))
    
    # Apply text cleaning, tokenization, stemming/lemmatization
    processed_texts = []
    processed_tokens = []
    
    for text in data['text']:
        cleaned_text, tokens = process_text(text, stem=stem, lemmatize=lemmatize)
        processed_texts.append(cleaned_text)
        processed_tokens.append(tokens)
    
    data['cleaned_text'] = processed_texts
    data['tokens'] = processed_tokens
    
    # Save preprocessed data
    data.to_csv(os.path.join(processed_data_path, output_file), index=False)
    logging.info(f'Saved processed data to {output_file}')

# Function to process a JSON file
def preprocess_json(input_file, output_file, stem=False, lemmatize=False):
    logging.info(f'Preprocessing JSON file: {input_file}')
    # Load raw data
    with open(os.path.join(raw_data_path, input_file), 'r') as f:
        data = json.load(f)
    
    processed_data = []
    
    for entry in data:
        cleaned_text, tokens = process_text(entry['text'], stem=stem, lemmatize=lemmatize)
        entry['cleaned_text'] = cleaned_text
        entry['tokens'] = tokens
        processed_data.append(entry)
    
    # Save preprocessed data
    with open(os.path.join(processed_data_path, output_file), 'w') as f:
        json.dump(processed_data, f, indent=4)
    logging.info(f'Saved processed data to {output_file}')

# Function to process a plain text file
def preprocess_txt(input_file, output_file, stem=False, lemmatize=False):
    logging.info(f'Preprocessing TXT file: {input_file}')
    # Load raw data
    with open(os.path.join(raw_data_path, input_file), 'r') as f:
        data = f.readlines()
    
    processed_texts = []
    
    for text in data:
        cleaned_text, tokens = process_text(text.strip(), stem=stem, lemmatize=lemmatize)
        processed_texts.append(cleaned_text)
    
    # Save preprocessed data
    with open(os.path.join(processed_data_path, output_file), 'w') as f:
        f.write('\n'.join(processed_texts))
    logging.info(f'Saved processed data to {output_file}')

# Function to preprocess data based on file extension
def preprocess_data(input_file, output_file, stem=False, lemmatize=False):
    logging.info(f'Preprocessing file: {input_file}')
    _, file_extension = os.path.splitext(input_file)
    
    if file_extension == '.csv':
        preprocess_csv(input_file, output_file, stem, lemmatize)
    elif file_extension == '.json':
        preprocess_json(input_file, output_file, stem, lemmatize)
    elif file_extension == '.txt':
        preprocess_txt(input_file, output_file, stem, lemmatize)
    else:
        logging.error(f'Unsupported file format: {file_extension}')

# Function to process an entire directory of raw data
def preprocess_directory(stem=False, lemmatize=False):
    logging.info('Preprocessing entire directory of raw data')
    for filename in os.listdir(raw_data_path):
        if filename.endswith('.csv') or filename.endswith('.json') or filename.endswith('.txt'):
            output_file = 'processed_' + filename
            preprocess_data(filename, output_file, stem, lemmatize)

# Entry point for script
if __name__ == "__main__":
    preprocess_directory(stem=True, lemmatize=True)