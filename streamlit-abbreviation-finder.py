import streamlit as st
import pandas as pd
import os
import nltk
from nltk import bigrams, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import chardet
import tempfile
import io

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def read_text(file):
    """
    Read a text file and return its contents as a string.
    """
    with open(file, 'rb') as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
    
    with open(file, 'r', encoding=encoding, errors='ignore') as file:
        text = file.read()
    return text

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
            if abbreviation not in english_stopwords:
                suggestions[word] = abbreviation

    # Process for common bigrams
    bi_gram_counts = Counter(bi_grams)
    for (w1, w2), count in bi_gram_counts.items():
        if count > 1 and all(word.isalpha() for word in [w1, w2]):
            abbreviation = w1[0].lower() + w2[0].lower()
            if abbreviation not in english_stopwords:
                suggestions[f"{w1} {w2}"] = abbreviation

    return suggestions

def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        text = read_text(tmp_file.name)
    words, bi_grams = tokenize_text(text)
    return suggest_abbreviations(words, bi_grams)

def convert_to_csv(suggestions):
    """
    Convert the suggestions dictionary to a CSV string.
    """
    df = pd.DataFrame(list(suggestions.items()), columns=['Original', 'Abbreviation'])
    # Sort the DataFrame by the 'Abbreviation' column
    df = df.sort_values(by='Abbreviation', ascending=False)
    return df.to_csv(index=False).encode('utf-8')

# Streamlit UI
st.title('Abbreviation Suggestion Tool')

uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
if uploaded_file is not None:
    suggestions = process_file(uploaded_file)
    if suggestions:
        st.write('Suggested Abbreviations:')
        # Convert suggestions to DataFrame for nicer display
        df = pd.DataFrame(list(suggestions.items()), columns=['Original', 'Abbreviation'])
        # Sort the DataFrame by the 'Abbreviation' column
        df = df.sort_values(by='Abbreviation', ascending=False)
        st.table(df)
        
        # CSV Download
        csv = convert_to_csv(suggestions)
        st.download_button(
            label="Download abbreviations as CSV",
            data=csv,
            file_name='abbreviations.csv',
            mime='text/csv',
        )
    else:
        st.write("No suggestions could be generated.")
