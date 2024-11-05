import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean the input text by removing unwanted characters and formatting.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    return text

def tokenize_text(text):
    """
    Tokenize the text into words.
    """
    # Tokenize the cleaned text
    tokens = word_tokenize(text)
    
    # Filter out tokens that are stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def lemmatize_tokens(tokens):
    """
    Lemmatize the tokens to their base form.
    """
    # Apply lemmatization on each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens

def preprocess_text(text):
    """
    Complete preprocessing pipeline: cleaning, tokenizing, and lemmatizing text.
    """
    # Clean the text
    text = clean_text(text)
    
    # Tokenize the text
    tokens = tokenize_text(text)
    
    # Lemmatize the tokens
    lemmatized_tokens = lemmatize_tokens(tokens)
    
    return lemmatized_tokens