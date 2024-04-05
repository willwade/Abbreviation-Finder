import streamlit as st
import pandas as pd
import nltk
from nltk import ngrams
from nltk.corpus import stopwords, words  # Corrected import here
from nltk.tokenize import word_tokenize
import chardet
import io
from collections import Counter
from collections import defaultdict
import textract
import tempfile
import xml.etree.ElementTree as ET

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')  # This ensures the 'words' corpus is available

english_words = set(words.words())  # This should now work without errors
english_stopwords = set(stopwords.words('english'))

# Modified read_text function using textract
def read_and_combine_texts(uploaded_files):
    combined_text = ""
    for uploaded_file in uploaded_files:
        # Process each file individually
        text = read_text(uploaded_file)
        combined_text += text + " "  # Add a space between texts from different files
    return combined_text


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
    
    # Check if the list is empty after filtering
    if not words:
        # Handle the case where no words meet the criteria
        # For example, return a placeholder or log a message
        return "NA"  # Placeholder abbreviation, adjust as needed
    
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
    
    # Return phrases and their frequencies
    return { ' '.join(phrase): count for phrase, count in common_phrases.items() if count >= min_frequency }

def process_text(text):
    # Find common phrases and their frequencies
    common_phrases_with_freq = find_common_phrases(text, max_length=7, min_frequency=2)
    
    # Generate abbreviations for common phrases
    existing_abbreviations = set()
    abbreviations = {}
    for phrase, freq in common_phrases_with_freq.items():
        abbreviation = unique_abbreviation(phrase, existing_abbreviations)
        existing_abbreviations.add(abbreviation)
        abbreviations[phrase] = (abbreviation, freq)
    
    # Process individual words not in common phrases
    words = set(word_tokenize(text.lower())) - set(' '.join(common_phrases_with_freq.keys()).split())
    for word in words:
        if word.lower() in english_stopwords or len(word) <= 1:
            continue
        if word not in abbreviations:  # Avoid reprocessing
            freq = text.lower().split().count(word)  # Simple frequency count for individual words
            abbreviation = unique_abbreviation(word, existing_abbreviations)
            existing_abbreviations.add(abbreviation)
            abbreviations[word] = (abbreviation, freq)
    
    return abbreviations


def create_df_and_sort(abbreviations):
    # Convert abbreviations and frequencies to a DataFrame
    df = pd.DataFrame([(original, abbr[0], abbr[1]) for original, abbr in abbreviations.items()],
                      columns=['Original', 'Abbreviation', 'Frequency'])
    # Sort by Frequency in descending order
    df = df.sort_values(by='Frequency', ascending=False)
    # Reset the index without adding the old index as a column
    df = df.reset_index(drop=True)
    return df

def filter_df(df, option):
    if option == 'Just Phrases':
        # Filter to show only phrases (assuming phrases contain spaces)
        return df[df['Original'].str.contains(' ')]
    elif option == 'Just Words':
        # Filter to show only words (assuming words do not contain spaces)
        return df[~df['Original'].str.contains(' ')]
    else:
        # 'All' option, no filtering needed
        return df

def calculate_savings(df, top_n):
    # Assume each word is 5 keystrokes on average
    avg_keystrokes_per_word = 5
    # Calculate keystroke savings for the top N abbreviations
    df['Keystroke Savings'] = df['Original'].apply(len) - df['Abbreviation'].apply(len)
    total_savings = df.head(top_n)['Keystroke Savings'].sum()
    # Calculate the average savings per word
    avg_savings_per_word = total_savings / top_n
    # Estimate the increase in "words" that could be typed per minute
    additional_words_per_minute = avg_savings_per_word / avg_keystrokes_per_word
    # Assuming an average typing speed of 40 WPM
    avg_typing_speed = 40
    # Calculate the percentage increase in WPM
    percentage_increase = (additional_words_per_minute / avg_typing_speed) * 100
    return total_savings, percentage_increase


def convert_to_csv(suggestions):
    """
    Convert the suggestions dictionary to a CSV string.
    """
    df = pd.DataFrame(list(suggestions.items()), columns=['Original', 'Abbreviation'])
    # Sort the DataFrame by the 'Abbreviation' column
    df = df.sort_values(by='Abbreviation', ascending=False)
    return df.to_csv(index=False).encode('utf-8')



def generate_plist_content(abbreviations):
    plist = ET.Element('plist', version="1.0")
    array = ET.SubElement(plist, 'array')
    
    for original, (abbreviation, _) in abbreviations.items():
        dict_elem = ET.SubElement(array, 'dict')
        phrase_key = ET.SubElement(dict_elem, 'key')
        phrase_key.text = 'phrase'
        phrase_value = ET.SubElement(dict_elem, 'string')
        phrase_value.text = original
        shortcut_key = ET.SubElement(dict_elem, 'key')
        shortcut_key.text = 'shortcut'
        shortcut_value = ET.SubElement(dict_elem, 'string')
        shortcut_value.text = abbreviation
    
    # Generate the XML string
    tree = ET.ElementTree(plist)
    xml_str = ET.tostring(plist, encoding='utf-8', method='xml').decode('utf-8')
    
    # Add XML declaration and DOCTYPE
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' \
              '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n' + xml_str
    
    return xml_str

# Streamlit UI code for uploading files and displaying results
st.title('Abbreviation Suggestion Tool')

uploaded_files = st.file_uploader("Choose text files", accept_multiple_files=True, type=['txt', 'docx', 'pdf', 'rtf', 'odt'])


if uploaded_files:
    combined_text = read_and_combine_texts(uploaded_files)
    suggestions = process_text(combined_text)  # Ensure process_text can handle the extracted text
    if suggestions:
        st.write('Suggested Abbreviations:')
        filter_option = st.selectbox(
            "Select items to display:",
            ('All', 'Just Phrases', 'Just Words'),
            index=0  # Default to showing 'All'
        )
        df = create_df_and_sort(suggestions)
        filtered_df = filter_df(df, filter_option)
        st.dataframe(filtered_df, hide_index=True)

        
        for top_n in [10, 50]:
            total_savings, percentage_increase = calculate_savings(filtered_df, top_n)
            st.write(f"By learning the top {top_n} abbreviations, you would save {total_savings} keystrokes, "
                     f"leading to an increase in WPM rate by approximately {percentage_increase:.2f}%.")
        
        # CSV Download
        csv = convert_to_csv(suggestions)
        st.download_button(
            label="Download abbreviations as CSV",
            data=csv,
            file_name='abbreviations.csv',
            mime='text/csv',
        )
        
        #Text replacements plist
        plist_content = generate_plist_content(suggestions)    
        # Offer the plist for download
        st.download_button(
            label="Download for Mac/iOS Text Replacements",
            data=plist_content,
            file_name='text_replacements.plist',
            mime='application/x-plist'
        )        
    else:
        st.write("No suggestions could be generated.")
