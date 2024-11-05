import pytest
import os
from data.scripts import preprocess, augment, split

@pytest.fixture(scope="module")
def raw_data():
    return [
        "I love this product!",
        "This is terrible, I hate it.",
        "Not bad, could be better.",
        "Amazing! Fantastic! Absolutely love it.",
        "Horrible experience, will not recommend."
    ]

@pytest.fixture(scope="module")
def processed_data():
    return [
        ["love", "product"],
        ["terrible", "hate"],
        ["bad", "better"],
        ["amazing", "fantastic", "love"],
        ["horrible", "experience", "not", "recommend"]
    ]

@pytest.fixture(scope="module")
def augmented_data():
    return [
        ["adore", "product"],
        ["awful", "hate"],
        ["poor", "better"],
        ["incredible", "fantastic", "adore"],
        ["dreadful", "experience", "not", "suggest"]
    ]

def test_data_preprocessing(raw_data, processed_data):
    preprocessed = [preprocess.preprocess_text(text) for text in raw_data]
    assert preprocessed == processed_data

def test_data_preprocessing_case_insensitive():
    # Test that preprocessing is case-insensitive
    raw_text = "I LOVE this Product!"
    processed = preprocess.preprocess_text(raw_text)
    assert processed == ["love", "product"]

def test_data_preprocessing_with_special_chars():
    # Test that preprocessing handles special characters correctly
    raw_text = "Amazing! This is an, excellent product?"
    processed = preprocess.preprocess_text(raw_text)
    assert processed == ["amazing", "excellent", "product"]

def test_data_preprocessing_with_numbers():
    # Test that preprocessing removes or handles numbers correctly
    raw_text = "This product costs 100 dollars."
    processed = preprocess.preprocess_text(raw_text)
    assert processed == ["product", "costs", "dollars"]

def test_data_preprocessing_empty_input():
    # Test preprocessing with empty input
    with pytest.raises(ValueError):
        preprocess.preprocess_text("")

def test_data_augmentation(processed_data, augmented_data):
    augmented = [augment.augment_text(tokens) for tokens in processed_data]
    assert augmented == augmented_data

def test_data_augmentation_with_edge_cases():
    # Test augmentation when no synonyms are available
    processed = ["love", "it"]
    augmented = augment.augment_text(processed)
    assert augmented == ["love", "it"]  # No synonyms, should return the same

def test_data_augmentation_empty_input():
    # Test augmentation with empty input
    with pytest.raises(ValueError):
        augment.augment_text([])

def test_data_splitting(raw_data):
    train, val, test = split.split_data(raw_data, train_size=0.7, val_size=0.15, test_size=0.15)
    
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1
    assert set(train + val + test) == set(raw_data)

def test_data_splitting_invalid_ratios():
    # Test invalid splitting ratios that don't sum up to 1
    with pytest.raises(ValueError):
        split.split_data([1, 2, 3], train_size=0.6, val_size=0.5, test_size=0.2)

def test_data_splitting_uneven_split():
    # Test data splitting with uneven division
    data = ["sample1", "sample2", "sample3", "sample4"]
    train, val, test = split.split_data(data, train_size=0.5, val_size=0.25, test_size=0.25)
    
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1

def test_data_splitting_small_dataset():
    # Test splitting a small dataset
    data = ["sample1", "sample2"]
    train, val, test = split.split_data(data, train_size=0.5, val_size=0.25, test_size=0.25)
    
    assert len(train) == 1
    assert len(val) == 0
    assert len(test) == 1

def test_data_splitting_empty_dataset():
    # Test splitting an empty dataset
    with pytest.raises(ValueError):
        split.split_data([], train_size=0.7, val_size=0.15, test_size=0.15)

def test_preprocessing_on_large_text():
    # Test how preprocessing handles very large text
    raw_text = " ".join(["great"] * 10000)  # Simulate a very large text
    processed = preprocess.preprocess_text(raw_text)
    assert processed == ["great"] * 10000

def test_augmentation_on_large_text():
    # Test augmentation on very large text
    processed = ["great"] * 10000
    augmented = augment.augment_text(processed)
    assert augmented == ["great"] * 10000 

def test_data_splitting_large_dataset():
    # Test splitting a large dataset
    data = ["sample"] * 10000
    train, val, test = split.split_data(data, train_size=0.7, val_size=0.15, test_size=0.15)
    
    assert len(train) == 7000
    assert len(val) == 1500
    assert len(test) == 1500

def test_preprocessing_unicode_text():
    # Test preprocessing on text with unicode characters
    raw_text = "I ❤️ this! Product 👍"
    processed = preprocess.preprocess_text(raw_text)
    assert processed == ["love", "product"]

def test_preprocessing_with_punctuation():
    # Test that preprocessing removes punctuation
    raw_text = "Great, awesome, amazing!"
    processed = preprocess.preprocess_text(raw_text)
    assert processed == ["great", "awesome", "amazing"]

def test_augmentation_with_punctuation():
    # Test augmentation on text with punctuation
    processed = ["great", "awesome", "amazing"]
    augmented = augment.augment_text(processed)
    assert augmented == ["great", "awesome", "amazing"] 

def test_augmentation_mixed_case():
    # Test that augmentation handles mixed-case words correctly
    processed = ["LOVE", "this", "Product"]
    augmented = augment.augment_text(processed)
    assert augmented == ["adore", "this", "product"]

def test_data_splitting_mixed_case_data():
    # Test data splitting with mixed-case text
    data = ["Sample1", "SAMPLE2", "sample3", "Sample4"]
    train, val, test = split.split_data(data, train_size=0.5, val_size=0.25, test_size=0.25)
    
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1

def test_data_preprocessing_special_case_sentences():
    # Test preprocessing on edge case sentences
    raw_texts = [
        "!!!",  # Just punctuation
        "1234567890",  # Numbers only
        "",  # Empty string
        "!!!!12345****"  # Mixed punctuation and numbers
    ]
    processed_texts = [
        preprocess.preprocess_text(text) for text in raw_texts
    ]
    assert processed_texts == [[], [], [], []]  # All cases should return empty lists