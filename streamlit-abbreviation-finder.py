import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
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
    raw_data = file.read()
    encoding = chardet.detect(raw_data)['encoding']
    text = raw_data.decode(encoding, errors='ignore')
    return text

def normalize_and_count(words):
    """
    Normalize words to lowercase for counting, but store the most common case (uppercase/lowercase) for each word.
    """
    lowercase_counts = Counter([word.lower() for word in words])
    case_counts = defaultdict(Counter)
    for word in words:
        case_counts[word.lower()][word] += 1
    # Choose the most common case for each word
    chosen_cases = {word: case_counts[word].most_common(1)[0][0] for word in lowercase_counts}
    return chosen_cases, lowercase_counts

def generate_unique_abbreviations(words):
    """
    Generate unique abbreviations for a list of words, ensuring no conflicts.
    """
    chosen_cases, counts = normalize_and_count(words)
    abbreviations = {}
    used_abbreviations = set()
    
    for word, _ in counts.most_common():  # Process words by frequency
        base_word = chosen_cases[word]
        abbreviation = base_word[:2].lower()
        # Ensure the abbreviation is unique
        original_abbreviation = abbreviation
        i = 2
        while abbreviation in used_abbreviations:
            i += 1
            abbreviation = (base_word[:i] + base_word[-1]).lower()[:2]  # Try a new abbreviation
            if i > len(base_word):  # Fallback if we run out of letters
                abbreviation = original_abbreviation + str(len(used_abbreviations))
        
        used_abbreviations.add(abbreviation)
        abbreviations[base_word] = abbreviation
    
    return abbreviations

def process_text(text):
    words = word_tokenize(text)
    # Filter out stopwords and non-alphabetic words
    words = [word for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
    return generate_unique_abbreviations(words)

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

uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'docx', 'rtf'])
if uploaded_file is not None:
    text = read_text(uploaded_file)
    suggestions = process_text(text)
    if suggestions:
        st.write('Suggested Abbreviations:')
        df = pd.DataFrame(list(suggestions.items()), columns=['Original', 'Abbreviation'])
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
