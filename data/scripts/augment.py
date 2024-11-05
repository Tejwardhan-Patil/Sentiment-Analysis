import random
import nltk
from nltk.corpus import wordnet
from textblob import TextBlob
from transformers import pipeline
import logging
import yaml
import os

nltk.download('wordnet')

# Logging setup
logging.basicConfig(filename='augment.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Configuration loading
def load_config(config_path='config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info("Configuration loaded successfully.")
        return config
    else:
        logger.error("Configuration file not found.")
        raise FileNotFoundError(f"Config file {config_path} not found")

# Synonym Replacement Function
def get_synonyms(word):
    try:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)
    except Exception as e:
        logger.error(f"Error finding synonyms for word '{word}': {e}")
        return []

def synonym_replacement(text, n):
    try:
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
        random.shuffle(random_word_list)
        
        num_replacements = min(n, len(random_word_list))
        
        for random_word in random_word_list[:num_replacements]:
            synonyms = get_synonyms(random_word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]

        logger.info(f"Synonym replacement completed for text: '{text}'")
        return ' '.join(new_words)
    except Exception as e:
        logger.error(f"Error in synonym replacement for text '{text}': {e}")
        return text

# Paraphrasing using transformers
def paraphrase_text(text):
    try:
        paraphraser = pipeline('text2text-generation', model="t5-base", task="paraphrase")
        result = paraphraser(text, max_length=100, num_return_sequences=1)
        logger.info(f"Paraphrased text generated: '{text}'")
        return result[0]['generated_text']
    except Exception as e:
        logger.error(f"Error paraphrasing text '{text}': {e}")
        return text

# Word Insertion Function
def random_word_insertion(text, n):
    try:
        words = text.split()
        for _ in range(n):
            new_word = random.choice(words)
            synonyms = get_synonyms(new_word)
            if synonyms:
                synonym = random.choice(synonyms)
                insert_position = random.randint(0, len(words)-1)
                words.insert(insert_position, synonym)
        
        logger.info(f"Random word insertion completed for text: '{text}'")
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error during word insertion for text '{text}': {e}")
        return text

# Random Swap Function
def random_swap(text, n):
    try:
        words = text.split()
        length = len(words)
        for _ in range(n):
            idx1, idx2 = random.sample(range(length), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        logger.info(f"Random word swap completed for text: '{text}'")
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error during random swap for text '{text}': {e}")
        return text

# Augmentation Pipeline
def augment_text(text, config):
    try:
        if config['augmentation']['synonym_replacement']:
            text = synonym_replacement(text, config['synonym_replacement']['num_replacements'])
        
        if config['augmentation']['random_insertion']:
            text = random_word_insertion(text, config['random_insertion']['num_insertions'])
        
        if config['augmentation']['random_swap']:
            text = random_swap(text, config['random_swap']['num_swaps'])
        
        if config['augmentation']['use_paraphrasing']:
            text = paraphrase_text(text)
        
        logger.info(f"Text augmentation completed for: '{text}'")
        return text
    except Exception as e:
        logger.error(f"Error in augmenting text '{text}': {e}")
        return text

# Preprocessing Function
def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Basic spell correction using TextBlob
        text_blob = TextBlob(text)
        corrected_text = str(text_blob.correct())
        logger.info(f"Preprocessing completed for text: '{text}'")
        return corrected_text
    except Exception as e:
        logger.error(f"Error preprocessing text '{text}': {e}")
        return text

# Augmentation with preprocessing
def augment_with_preprocessing(text, config):
    preprocessed_text = preprocess_text(text)
    augmented_text = augment_text(preprocessed_text, config)
    return augmented_text

# Augmenting batch of data
def augment_batch(texts, config):
    augmented_texts = []
    for text in texts:
        augmented_text = augment_with_preprocessing(text, config)
        augmented_texts.append(augmented_text)
    return augmented_texts

# Usage
if __name__ == "__main__":
    # Load configuration
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Data
    text_data = [
        "This is a wonderful day to learn something new.",
        "Data science is an evolving field.",
        "Artificial Intelligence is transforming the world.",
        "Python is a popular programming language for machine learning."
    ]
    
    augmented_data = augment_batch(text_data, config)
    
    # Output augmented data
    for i, original in enumerate(text_data):
        logger.info(f"Original Text: {original}")
        logger.info(f"Augmented Text: {augmented_data[i]}")
        print(f"Original: {original}")
        print(f"Augmented: {augmented_data[i]}\n")