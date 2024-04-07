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
import re

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


def heuristic_syllables(word):
    # This is a placeholder for your syllable detection logic
    syllables = re.findall(r'[bcdfghjklmnpqrstvwxyz]*[aeiouy]+', word.lower())
    return [syl[0] for syl in syllables]
    
def generate_syllable_abbreviation(phrase):
    words = phrase.split()
    abbreviation = ''
    for word in words:
        syllables = heuristic_syllables(word)
        # Take the first letter of each syllable for the abbreviation
        for syllable in syllables:
            abbreviation += syllable[0]
    return abbreviation

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
    words = [word for word in word_tokenize(word_or_phrase) if word.lower() not in english_stopwords and len(word) > 1]
    
    if len(words) == 1:  # It's a single word, use syllables
        word = words[0]
        syllable_initials = heuristic_syllables(word)
        abbreviation = ''.join(syllable_initials)
    else:  # It's a phrase, use initial letters
        abbreviation = ''.join(word[0] for word in words).lower()
    
    return abbreviation

def unique_abbreviation(original, existing_abbreviations, english_words, avoid_numbers=False):
    abbreviation = generate_abbreviation(original)
    base_abbreviation = abbreviation
    counter = 1
    
    while abbreviation in existing_abbreviations or abbreviation in english_words:
        if avoid_numbers:
            # Attempt to modify the abbreviation without using numbers
            if len(base_abbreviation) >= counter:
                # Add an additional letter from the original word or repeat the last letter
                next_letter = base_abbreviation[counter % len(base_abbreviation)]
                abbreviation += next_letter
            else:
                # Fallback to repeating the last letter if no more unique letters are available
                abbreviation += base_abbreviation[-1]
        else:
            # Use numbers to ensure uniqueness
            abbreviation = f"{base_abbreviation}{counter}"
        
        counter += 1
    
    existing_abbreviations.add(abbreviation)
    return abbreviation

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
        abbreviation = unique_abbreviation(phrase, existing_abbreviations, english_words, avoid_numbers)
        existing_abbreviations.add(abbreviation)
        abbreviations[phrase] = (abbreviation, freq)
    
    # Process individual words not in common phrases
    words = set(word_tokenize(text.lower())) - set(' '.join(common_phrases_with_freq.keys()).split())
    for word in words:
        if word.lower() in english_stopwords or len(word) <= 1:
            continue
        if word not in abbreviations:  # Avoid reprocessing
            freq = text.lower().split().count(word)  # Simple frequency count for individual words
            abbreviation = unique_abbreviation(word, existing_abbreviations, english_words)
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


def convert_to_csv(df):
    """
    Convert the filtered DataFrame to a CSV string.
    """
    # Sort the DataFrame by the 'Abbreviation' column if needed
    df = df.sort_values(by='Abbreviation', ascending=False)
    return df.to_csv(index=False).encode('utf-8')


def generate_plist_content(df):
    plist = ET.Element('plist', version="1.0")
    array = ET.SubElement(plist, 'array')
    
    for _, row in df.iterrows():
        dict_elem = ET.SubElement(array, 'dict')
        phrase_key = ET.SubElement(dict_elem, 'key')
        phrase_key.text = 'phrase'
        phrase_value = ET.SubElement(dict_elem, 'string')
        phrase_value.text = row['Original']
        shortcut_key = ET.SubElement(dict_elem, 'key')
        shortcut_key.text = 'shortcut'
        shortcut_value = ET.SubElement(dict_elem, 'string')
        shortcut_value.text = row['Abbreviation']
    
    tree = ET.ElementTree(plist)
    xml_str = ET.tostring(plist, encoding='utf-8', method='xml').decode('utf-8')
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' \
              '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n' + xml_str
    
    return xml_str

# For this I really want to offer the option of adding to this 
# https://www.autohotkey.com/boards/viewtopic.php?f=83&t=120220&start=60#p565896

def generate_ahk_script(df):
    script_lines = []
    for _, row in df.iterrows():
        script_line = f"::{row['Abbreviation']}::{row['Original']}"
        script_lines.append(script_line)
    script_lines.append(":?*:altion::lation")
    return "\n".join(script_lines)

# Streamlit UI code for uploading files and displaying results
st.title('Abbreviation Suggestion Tool')
st.markdown("""
    This tool helps you generate abbreviations for long words or phrases, making your typing faster and more efficient. 
    Upload your documents, and the tool will analyze the text to suggest useful abbreviations. Use the filters to include your likely abbreviations based on frequency found in your text. 
    You can then download these abbreviations in CSV format, a plist file for [Mac/iOS text replacements](https://support.apple.com/en-gb/guide/mac-help/mchl2a7bd795/mac) or as a [autohotkey](https://www.autohotkey.com) file.
    Want more ideas why abbreviations might be useful? Have a read of [this](https://blog.abreevy8.io/you-dont-have-to-type-faster-to-type-faster/). Bear in mind though the cognitive effort to learn these abbreviations. 
    **NB: We don't save your uploaded documents - we just parse them then display the summarised data here**
""")
uploaded_files = st.file_uploader("Choose text files", accept_multiple_files=True, type=['txt', 'docx', 'pdf', 'rtf', 'odt'])
avoid_numbers = st.checkbox("No numbers in abbreviations", value=False)

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
        min_frequency = st.slider("Minimum frequency", min_value=1, max_value=10, value=1)

        df = create_df_and_sort(suggestions)
        filtered_df = filter_df(df, filter_option)
        # Assuming 'filtered_df' is the DataFrame after applying the first filter (phrases/words)
        filtered_df = filtered_df[filtered_df['Frequency'] >= min_frequency]
        st.dataframe(filtered_df, hide_index=True)

        
        for top_n in [10, 50]:
            total_savings, percentage_increase = calculate_savings(filtered_df, top_n)
            st.write(f"By learning the top {top_n} abbreviations, you would save {total_savings} keystrokes, "
                     f"leading to an increase in WPM rate by approximately {percentage_increase:.2f}%.")
        
        if filtered_df is not None and not filtered_df.empty:
            # CSV Download
            csv = convert_to_csv(filtered_df)
            st.download_button(
                label="⊞  Download abbreviations as CSV",
                data=csv,
                file_name='abbreviations.csv',
                mime='text/csv',
            )
            
            # Text replacements plist
            plist_content = generate_plist_content(filtered_df)    
            # Offer the plist for download
            st.download_button(
                label=" Download for Mac/iOS Text Replacements",
                data=plist_content,
                file_name='Text Substitutions.plist',
                mime='application/x-plist'
            )
            ahk_script_content = generate_ahk_script(filtered_df)
            st.download_button(
                label="⊞ Download as AutoHotkey Script",
                data=ahk_script_content,
                file_name='abbreviations.ahk',
                mime='text/plain'
            )

    else:
        st.write("No suggestions could be generated.")
