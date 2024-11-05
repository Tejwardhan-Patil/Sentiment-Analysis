library(tm)    
library(SnowballC)   
library(textstem)    
library(dplyr)      

# Function to convert text to lowercase
to_lowercase <- function(text) {
  tolower(text)
}

# Function to remove punctuation
remove_punctuation <- function(text) {
  gsub("[[:punct:]]+", "", text)
}

# Function to remove numbers
remove_numbers <- function(text) {
  gsub("[[:digit:]]+", "", text)
}

# Function to remove stopwords
remove_stopwords <- function(text) {
  stopwords_list <- stopwords("en")
  words <- unlist(strsplit(text, "\\s+"))
  filtered_words <- words[!words %in% stopwords_list]
  paste(filtered_words, collapse = " ")
}

# Function to apply stemming
apply_stemming <- function(text) {
  words <- unlist(strsplit(text, "\\s+"))
  stemmed_words <- wordStem(words, language = "en")
  paste(stemmed_words, collapse = " ")
}

# Function to apply lemmatization
apply_lemmatization <- function(text) {
  lemmatize_strings(text)
}

# Function to tokenize text
tokenize_text <- function(text) {
  words <- unlist(strsplit(text, "\\s+"))
  return(words)
}

# Master function for text preprocessing
preprocess_text <- function(text) {
  text <- to_lowercase(text)
  text <- remove_punctuation(text)
  text <- remove_numbers(text)
  text <- remove_stopwords(text)
  text <- apply_lemmatization(text)  
  return(text)
}