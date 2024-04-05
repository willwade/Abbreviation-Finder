import streamlit as st
import pandas as pd
import nltk
from nltk import ngrams
from nltk.corpus import stopwords, words  # Corrected import here
from nltk.tokenize import word_tokenize
import chardet
import io
from collections import Counter
import textract

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')  # This ensures the 'words' corpus is available

english_words = set(words.words())  # This should now work without errors
english_stopwords = set(stopwords.words('english'))

# Modified read_text function using textract
def read_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    text = textract.process(tmp_file_path, encoding='utf-8')
    return text.decode('utf-8')


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


def is_real_word(abbreviation):
    return abbreviation.lower() in english_words

def next_vowel(word):
    for letter in word[1:]:
        if letter.lower() in 'aeiou':
            return letter
    return ''

def generate_abbreviation(word_or_phrase):
    # Remove small words for phrases
    words = [word for word in word_tokenize(word_or_phrase) if word.lower() not in english_stopwords and len(word) > 1]
    if len(words) > 1:  # It's a phrase
        abbreviation = ''.join(word[0] for word in words).lower()
    else:  # It's a single word
        word = words[0]
        abbreviation = (word[0] + word[1]).lower() if len(word) > 1 else word[0].lower()

    return abbreviation

def unique_abbreviation(original, existing_abbreviations):
    abbreviation = generate_abbreviation(original)
    if abbreviation not in existing_abbreviations and not is_real_word(abbreviation):
        return abbreviation

    # Try first letter + next vowel
    if len(original) > 2:
        abbreviation = (original[0] + next_vowel(original)).lower()
        if abbreviation not in existing_abbreviations and not is_real_word(abbreviation):
            return abbreviation

    # Append numbers (1-9), then double digits as last case
    for i in range(1, 100):
        new_abbreviation = f"{abbreviation}{i}"
        if new_abbreviation not in existing_abbreviations:
            return new_abbreviation

def find_common_phrases(text, max_length=7, min_frequency=2):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    common_phrases = Counter()
    for n in range(2, max_length + 1):
        for n_gram in ngrams(tokens, n):
            if all(word not in stop_words for word in n_gram):  # Ensure n-gram doesn't consist only of stopwords
                common_phrases[n_gram] += 1
    
    # Filter by frequency and return phrases
    return [' '.join(phrase) for phrase, count in common_phrases.items() if count >= min_frequency]

def process_text(text):
    # First, find common phrases in the text
    common_phrases = find_common_phrases(text, max_length=7, min_frequency=2)
    
    # Generate abbreviations for common phrases
    existing_abbreviations = set()
    abbreviations = {}
    for phrase in common_phrases:
        abbreviation = unique_abbreviation(phrase, existing_abbreviations)
        existing_abbreviations.add(abbreviation)
        abbreviations[phrase] = abbreviation
    
    # Now, process individual words that are not part of any common phrases
    words = set(word_tokenize(text)) - set(' '.join(common_phrases).split())
    for word in words:
        if word.lower() in english_stopwords or len(word) <= 1:
            continue
        if word not in abbreviations:  # Avoid processing words that are already handled as part of phrases
            abbreviation = unique_abbreviation(word, existing_abbreviations)
            existing_abbreviations.add(abbreviation)
            abbreviations[word] = abbreviation
    
    return abbreviations


def convert_to_csv(suggestions):
    """
    Convert the suggestions dictionary to a CSV string.
    """
    df = pd.DataFrame(list(suggestions.items()), columns=['Original', 'Abbreviation'])
    # Sort the DataFrame by the 'Abbreviation' column
    df = df.sort_values(by='Abbreviation', ascending=False)
    return df.to_csv(index=False).encode('utf-8')

# Streamlit UI code for uploading files and displaying results
st.title('Abbreviation Suggestion Tool')

uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'docx', 'rtf', 'pdf', 'odt'])
if uploaded_file is not None:
    text = read_text(uploaded_file)
    suggestions = process_text(text)  # Ensure process_text can handle the extracted text
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