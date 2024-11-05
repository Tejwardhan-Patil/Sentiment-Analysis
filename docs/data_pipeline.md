# Data Pipeline

The data pipeline for the sentiment analysis project is designed to handle the collection, preprocessing, augmentation, and splitting of text data. This process ensures that the data is ready for model training and evaluation.

## 1. Data Collection

- **Directory**: `data/raw/`
- Raw text data collected from various sources, such as social media, reviews, and articles, are stored here.

## 2. Preprocessing

- **File**: `data/scripts/preprocess.py`
- The preprocessing script handles tasks such as text cleaning, tokenization, stopword removal, and other text normalization processes.
- Output: Processed text data stored in `data/processed/`.

## 3. Data Augmentation

- **File**: `data/scripts/augment.py`
- This script augments the data by applying techniques like synonym replacement and paraphrasing to increase the size and variability of the training data.

## 4. Data Splitting

- **File**: `data/scripts/split.py`
- The dataset is split into training, validation, and test sets to ensure proper evaluation and model tuning.
- Output: Data subsets are stored in `data/split/`.

## 5. Exploratory Data Analysis (EDA)

- **File**: `data/scripts/eda.Rmd`
- A script that utilizes R's statistical analysis and visualization capabilities to explore the raw and processed data.
