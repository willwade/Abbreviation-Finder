import os
import nltk
from nltk import bigrams, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import argparse
import chardet
import string

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def read_texts(folder_path):
    """
    Read all text files in the specified folder and return their contents as a single string.
    """
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                texts.append(file.read())
    return ' '.join(texts)

def tokenize_text(text):
    """
    Tokenize the given text into words and bigrams.
    """
    words = word_tokenize(text)
    bi_grams = list(bigrams(words))
    return words, bi_grams

def suggest_abbreviations(words, bi_grams):
    """
    Suggest abbreviations for long words and common bigrams.
    """
    suggestions = {}
    english_stopwords = stopwords.words('english')

    # Filter for long and complex words
    for word in set(words):
        if len(word) > 7 and word.lower() not in english_stopwords and word.isalpha():
            abbreviation = word[:2].lower()  # Use first two letters, lowercase
            # Ensure abbreviation is not a whole word or a common English word
            if abbreviation not in english_stopwords:
                suggestions[word] = abbreviation

    # Process for common bigrams
    bi_gram_counts = Counter(bi_grams)
    for (w1, w2), count in bi_gram_counts.items():
        if count > 1 and all(word.isalpha() for word in [w1, w2]):  # Ensure both words are purely alphabetic
            abbreviation = w1[0].lower() + w2[0].lower()  # First letter of each word, lowercase
            # Ensure abbreviation is not a whole word or a common English word
            if abbreviation not in english_stopwords:
                suggestions[f"{w1} {w2}"] = abbreviation

    return suggestions

def main(folder_path):
    text = read_texts(folder_path)
    words, bi_grams = tokenize_text(text)
    suggestions = suggest_abbreviations(words, bi_grams)

    for original, abbreviation in suggestions.items():
        print(f"{original} -> {abbreviation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate abbreviation suggestions from text files in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing text files.")
    args = parser.parse_args()

    main(args.folder_path)
