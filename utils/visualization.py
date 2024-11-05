import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_training_curves(history, metrics=['loss', 'accuracy']):
    """
    Plots the training and validation curves for specified metrics.
    
    Args:
        history: Training history object containing metric data.
        metrics: List of metrics to plot (e.g., 'loss', 'accuracy').
    """
    epochs = range(1, len(history.history[metrics[0]]) + 1)
    
    for metric in metrics:
        train_values = history.history[metric]
        val_values = history.history[f'val_{metric}']
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_values, 'bo-', label=f'Training {metric}')
        plt.plot(epochs, val_values, 'r*-', label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

def generate_word_cloud(text, max_words=100, colormap='viridis'):
    """
    Generates and displays a word cloud from a given text.
    
    Args:
        text: Input text data for generating the word cloud.
        max_words: Maximum number of words to include in the word cloud.
        colormap: Color map for the word cloud visualization.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=max_words, colormap=colormap).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def plot_confusion_matrix(confusion_matrix, class_names):
    """
    Plots a confusion matrix heatmap.
    
    Args:
        confusion_matrix: Confusion matrix to visualize.
        class_names: List of class names to annotate the matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()