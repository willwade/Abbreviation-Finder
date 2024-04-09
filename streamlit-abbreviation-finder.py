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

# setup layouts


# 1. qwerty staggered
keyboard_layout = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', 'g', 'h', 'n', ' '],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k', 'l', ' '],
    'n': ['b', 'h', 'j', 'm', ' '],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'f', 'g', 'b', ' '],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x']
}
# 2. AEIOU non-staggered
aeiou_layout = {
    'a': ['e', 'b', 'f'],
    'b': ['a', 'e', 'f', 'c', 'g'],
    'c': ['b', 'f', 'g', 'd', 'h'],
    'd': ['c', 'g', 'h'],
    'e': ['a', 'b', 'f', 'j', 'i'],
    'f': ['a', 'b', 'e', 'c', 'g', 'k', 'i', 'j'],
    'g': ['b', 'c', 'f', 'd', 'h', 'l', 'm'],
    'h': ['c', 'd', 'g', 'i', 'n', 'm'],
    'i': ['f', 'e', 'j', 'p', 'o'],
    'j': ['e','f', 'g', 'k', 'q', 'p', 'o','i'],
    'k': ['j', 'f', 'g', 'h', 'l','r','q','p'],
    'l': ['h', 'g', 'k', 'm', 'p', 'q','s'],
    'm': ['l', 'n', 'r', 's', 't'],
    'n': ['m', 's', 't'],
    'o': ['i', 'j', 'p', 'v', 'u'],
    'p': ['i', 'j', 'k', 'o', 'q', 'u','v','w'],
    'q': ['j', 'k', 'l', 'p', 'r', 'v','w','x'],
    'r': ['k', 'l', 'm', 'q', 's', 'w','x','y'],
    's': ['l', 'm', 'n', 'r', 't', 'x','y','z'],
    't': ['m', 'n', 's', 'y', 'z'],
    'u': ['o', 'p', 'v'],
    'v': ['o', 'p', 'q', 'u', 'w'],
    'w': ['p', 'q', 'r', 'v', 'x',' '],
    'x': ['q', 'r', 'w','s','y'],
    'y': ['t', 'u', 'x', 'z'],
    'z': ['s', 't', 'y']
}
#3. ABC non staggered
abc_layout = {
    'a': ['b', 'k'],
    'b': ['a', 'k', 'l', 'c'],
    'c': ['b', 'l', 'm', 'd'],
    'd': ['c', 'm', 'n', 'e'],
    'e': ['d', 'n', 'o', 'f'],
    'f': ['e', 'o', 'p', 'g'],
    'g': ['f', 'p', 'q', 'h'],
    'h': ['g', 'q', 'r', 'i'],
    'i': ['h', 'r', 's', 'j'],
    'j': ['i', 's', 't'],
    'k': ['a', 'b', 'l', 't'],
    'l': ['k', 'b', 'c', 'm', 'u', 't'],
    'm': ['l', 'c', 'd', 'n', 'v', 'w','u'],
    'n': ['m', 'd', 'e', 'o', 'w', 'v'],
    'o': ['n', 'e', 'f', 'p', 'x', 'w'],
    'p': ['o', 'f', 'g', 'q', 'y', 'x'],
    'q': ['p', 'g', 'h', 'r', 'y', 'z'],
    'r': ['q', 'h', 'i', 's',',','z'],
    's': ['r', 'i', 'j', 't'],
    't': ['s', 'j', 'u'],
    'u': ['t', 'k', 'l', 'u'],
    'v': ['m', 'u', 'n', 'w',' '],
    'w': ['n', 'v','o', 'x',' '],
    'x': ['o','p', 'w', 'y',' '],
    'y': ['p','q', 'x', 'z',' '],
    'z': ['r', 'y','q',',',' ','!']
}
#4.  Row-col scanning for AEIOU
blocked_aeiou_layout = {
    'a': ['b'],
    'b': ['a', 'c'],
    'c': ['b', 'd'],
    'd': ['c'],
    'e': ['i'],
    'f': ['e', 'g'],
    'g': ['f', 'h'],
    'h': ['g', 'i'],
    'i': ['j'],
    'j': ['i', 'k'],
    'k': ['j', 'l'],
    'l': ['k', 'm'],
    'm': ['l', 'n'],
    'n': ['m'],
    'o': ['p'],
    'p': ['o', 'q'],
    'q': ['p', 'r'],
    'r': ['q', 's'],
    's': ['r', 't'],
    't': ['s'],
    'u': ['v'],
    'v': ['u', 'w'],
    'w': ['v', 'x'],
    'x': ['w','y'],
    'y': ['x','z'],
    'z': ['y']
}

#5.  Linear A-Z layout. Assume space is at end. This could be v wrong
linear_layout = {chr(97 + i): [chr(97 + max(0, i - 1)), chr(97 + min(25, i + 1))] for i in range(26)}
linear_layout['z'].append(' ')
linear_layout[' '] = ['z']

#6. Frequency linear
frequency_order = 'eisatnrolcdpumghyfbvwkxjq'
linear_frequency_layout = {}

for i, letter in enumerate(frequency_order):
    # For each letter, find the previous and next letters in the frequency order
    prev_letter = frequency_order[i - 1] if i - 1 >= 0 else None
    next_letter = frequency_order[i + 1] if i + 1 < len(frequency_order) else None

    # Initialize the list of adjacent letters
    adjacent_letters = []

    if prev_letter is not None:
        adjacent_letters.append(prev_letter)
    if next_letter is not None:
        adjacent_letters.append(next_letter)

    # Assign the list of adjacent letters to the current letter in the layout
    linear_frequency_layout[letter] = adjacent_letters

#7. Row/col frequency
rows = [
[' ', 'e', 'a', 'r', 'd', 'u'],
['t', 'o', 'i', 'l', 'g', 'v'],
['n', 's', 'f', 'y', 'x', '.'],
['h', 'c', 'p', 'k', 'j', "'"],
['m', 'b', 'w', 'q', 'z', '?']
]

#8 Row-column frequency layout
row_column_frequency_layout = {}

for row in rows:
    for i, letter in enumerate(row):
        # For each letter, find the previous and next letters in the row
        prev_letter = row[i - 1] if i - 1 >= 0 else None
        next_letter = row[i + 1] if i + 1 < len(row) else None

        # Initialize the list of adjacent letters
        adjacent_letters = []

        if prev_letter is not None:
            adjacent_letters.append(prev_letter)
        if next_letter is not None:
            adjacent_letters.append(next_letter)

        # Assign the list of adjacent letters to the current letter in the layout
        row_column_frequency_layout[letter] = adjacent_letters



layout_mapping = {
    'QWERTY Staggered': keyboard_layout,
    'AEIOU Non-Staggered': aeiou_layout,
    'ABC Non-Staggered': abc_layout,
    'Row-Col Scanning for AEIOU': blocked_aeiou_layout,
    'Linear A-Z Layout': linear_layout,
    'Linear Frequency Layout': linear_frequency_layout,
    'Row/Col Frequency Layout': row_column_frequency_layout
}

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
    return [syl[0] for syl in syllables] if syllables else []
    
def generate_syllable_abbreviation(phrase, existing_abbreviations, prepend=False):
    words = phrase.split()
    abbreviation = ''
    for word in words:
        syllables = heuristic_syllables(word)
        if syllables:
            # Use up to the first two characters of the first syllable
            abbreviation += syllables[0][:2]
        else:
            # Fallback if no syllables are found
            abbreviation += word[:2].lower()

    # Ensure uniqueness and adjust if necessary
    original_abbreviation = abbreviation
    counter = 1
    while abbreviation in existing_abbreviations or (not prepend and is_real_word(abbreviation)):
        if len(abbreviation) >= len(phrase):
            # Ensure abbreviation doesn't exceed the original phrase length
            abbreviation = original_abbreviation[:max(1, len(phrase)-1)]  # Ensure at least 1 character
        new_abbreviation = abbreviation + (phrase[counter] if counter < len(phrase) else str(counter))
        if new_abbreviation not in existing_abbreviations and (not is_real_word(new_abbreviation) or prepend):
            existing_abbreviations.add(new_abbreviation)
            return new_abbreviation
        counter += 1
        if counter > 100:  # Prevent infinite loops
            break

    existing_abbreviations.add(abbreviation)
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

def generate_proximity_based_abbreviation(word_or_phrase, layout, existing_abbreviations, prepend):
    # Start with the standard abbreviation logic
    if ' ' in word_or_phrase:
        abbreviation = ''.join(word[0] for word in word_or_phrase.split()).lower()
    else:
        # Single word, attempt to use syllables
        syllable_initials = heuristic_syllables(word_or_phrase)
        abbreviation = ''.join(syllable_initials)

    # If prepend is True, skip the length and english_words check
    if prepend:
        if abbreviation not in existing_abbreviations:
            existing_abbreviations.add(abbreviation)
            return abbreviation
    else:
        # Ensure the abbreviation is at least two letters long by using layout neighbors if needed
        if len(abbreviation) < 2:  # Adjust for the length of the prefix
            first_letter = word_or_phrase[0].lower()
            if first_letter in layout and len(word_or_phrase) > 1:
                neighbors = layout[first_letter]
                second_letter_candidates = [neighbor for neighbor in neighbors if neighbor in word_or_phrase[1:]]
                if second_letter_candidates:
                    abbreviation += second_letter_candidates[0]
                else:
                    abbreviation += word_or_phrase[1].lower()
            else:
                abbreviation += first_letter

    original_abbreviation = abbreviation
    counter = 1
    while abbreviation in existing_abbreviations or (not prepend and abbreviation in english_words):
        last_letter = abbreviation[-1]
        if last_letter in layout:
            for neighbor in layout[last_letter]:
                potential_abbreviation = original_abbreviation + neighbor
                if potential_abbreviation not in existing_abbreviations and (prepend or potential_abbreviation not in english_words):
                    abbreviation = potential_abbreviation
                    break
            else:
                abbreviation = original_abbreviation + str(counter)
                counter += 1
        else:
            abbreviation = original_abbreviation + str(counter)
            counter += 1

        if counter > 100:  # Prevent infinite loops
            break

    existing_abbreviations.add(abbreviation)
    return abbreviation

def generate_abbreviation(word_or_phrase, existing_abbreviations, prepend_option=False):
    if ' ' in word_or_phrase:  # It's a phrase
        abbreviation = ''.join(word[0] for word in word_or_phrase.split()).lower()
    else:  # It's a single word
        syllable_initials = heuristic_syllables(word_or_phrase)
        abbreviation = ''.join(syllable_initials)

    # Initial checks for abbreviation length and uniqueness
    if abbreviation not in existing_abbreviations and (not is_real_word(abbreviation) or prepend_option) and len(abbreviation) < len(word_or_phrase):
        existing_abbreviations.add(abbreviation)
        return abbreviation

    # Adjust abbreviation to ensure uniqueness and shorter than the original word
    counter = 1
    while True:
        if len(abbreviation) >= len(word_or_phrase):
            # Ensure abbreviation doesn't exceed the original word length
            abbreviation = abbreviation[:len(word_or_phrase)-1]
        new_abbreviation = abbreviation + (word_or_phrase[counter] if counter < len(word_or_phrase) else str(counter))
        if new_abbreviation not in existing_abbreviations and (not is_real_word(new_abbreviation) or prepend_option) and len(new_abbreviation) < len(word_or_phrase):
            existing_abbreviations.add(new_abbreviation)
            return new_abbreviation
        counter += 1
        if counter > 100:  # Prevent infinite loops
            break

def generate_all_abbreviations(text, layouts, prepend=False):
    # Tokenize the text and filter out stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english') and token.isalpha()]

    # Calculate the frequency of each token
    token_frequency = Counter(filtered_tokens)

    # Find common phrases and calculate their frequencies
    common_phrases_with_freq = find_common_phrases(text, max_length=7, min_frequency=2)
    # Combine token and phrase frequencies
    all_frequencies = {**common_phrases_with_freq, **token_frequency}

    abbreviation_variants = []

    for original, freq in all_frequencies.items():
        if len(original) <= 1 or original in stopwords.words('english'):
            continue

        # Reset existing abbreviations for each type
        existing_standard_abbr = set()
        standard_abbr = generate_abbreviation(original, existing_standard_abbr, prepend)

        existing_syllable_abbr = set()
        syllable_abbr = generate_syllable_abbreviation(original, existing_syllable_abbr,prepend) if ' ' not in original else standard_abbr

        # Generate proximity-based abbreviations for each layout, with separate tracking
        proximity_abbrs = {}
        for layout_name, layout in layouts.items():
            existing_proximity_abbr = set()
            proximity_abbrs[layout_name] = generate_proximity_based_abbreviation(original, layout, existing_proximity_abbr,prepend)

        abbreviation_variants.append({
            'Original': original,
            'Frequency': freq,
            'Standard': standard_abbr,
            'Syllable': syllable_abbr,
            **proximity_abbrs
        })

    return pd.DataFrame(abbreviation_variants)
    
def unique_abbreviation(original, existing_abbreviations, english_words, avoid_numbers=False):
    abbreviation = generate_abbreviation(original,existing_abbreviations)
    base_abbreviation = abbreviation
    counter = 1

    while abbreviation in existing_abbreviations or abbreviation in english_words:
        if avoid_numbers:
            # Generate a new abbreviation variant by appending a new letter or duplicating the last one
            # This checks if we've tried less than the length of the original word to find a unique letter
            if counter <= len(original):
                next_index = counter % len(original)  # Use modulo to cycle through the original word's letters
                next_letter = original[next_index]
            else:
                # If all letters have been tried, start doubling the last letter of the abbreviation
                next_letter = abbreviation[-1]
                
            abbreviation = base_abbreviation + next_letter
        else:
            # Use numbers to ensure uniqueness
            abbreviation = f"{base_abbreviation}{counter}"
        
        counter += 1
        if counter > 100:  # Safety check to prevent infinite loops
            break

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

def process_text(text,avoid_numbers=False):
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
            abbreviation = unique_abbreviation(word, existing_abbreviations, english_words,avoid_numbers)
            existing_abbreviations.add(abbreviation)
            abbreviations[word] = (abbreviation, freq)
    
    return abbreviations


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

def adjust_abbreviations_for_prepend(df, prepend_option):
    # Iterate over each abbreviation type column (excluding 'Original' and 'Frequency')
    for col in df.columns.drop(['Original', 'Frequency']):
        # Create a new column for adjusted abbreviations
        new_col_name = f"Adjusted {col}"
        
        if prepend_option:
            # Apply the prepend logic
            df[new_col_name] = df[col].apply(lambda x: '\\' + x if len(x) < 2 else x)
        else:
            # If prepend option is not selected, just copy the original abbreviations
            df[new_col_name] = df[col]
    
    return df


def calculate_savings(df, top_n, abbreviation_column, user_typing_speed):
    # Assume each word is 5 keystrokes on average
    avg_keystrokes_per_word = 5
    # Calculate keystroke savings for the top N abbreviations
    df = df.copy()
    df.loc[:, 'Keystroke Savings'] = df['Original'].apply(lambda x: len(x) if x is not None else 0) - df[abbreviation_column].apply(lambda x: len(x) if x is not None else 0)
    total_savings = df.head(top_n)['Keystroke Savings'].sum()
    # Calculate the average savings per word
    avg_savings_per_word = total_savings / top_n
    # Estimate the increase in "words" that could be typed per minute
    additional_words_per_minute = avg_savings_per_word / avg_keystrokes_per_word
    # Use the user's actual typing speed instead of assuming an average typing speed of 40 WPM
    # Calculate the percentage increase in WPM
    percentage_increase = (additional_words_per_minute / user_typing_speed) * 100
    return total_savings, percentage_increase


def convert_to_csv(df, abbreviation_column):
    """
    Convert the filtered DataFrame to a CSV string, sorting by the specified abbreviation column.
    """
    df = df.sort_values(by=abbreviation_column, ascending=False)
    return df.to_csv(index=False).encode('utf-8')


def generate_plist_content(df, abbreviation_column):
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
        shortcut_value.text = row[abbreviation_column]  # Use dynamic column name
    
    tree = ET.ElementTree(plist)
    xml_str = ET.tostring(plist, encoding='utf-8', method='xml').decode('utf-8')
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' \
              '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n' + xml_str
    
    return xml_str


# For this I really want to offer the option of adding to this 
# https://www.autohotkey.com/boards/viewtopic.php?f=83&t=120220&start=60#p565896

def generate_ahk_script(df, abbreviation_column):
    script_lines = []
    for _, row in df.iterrows():
        # Use the dynamic column name to access the abbreviation
        script_line = f"::{row[abbreviation_column]}::{row['Original']}"
        script_lines.append(script_line)
    # Assuming ":?*:altion::lation" is a static example or placeholder, you might adjust or remove it
    script_lines.append(":?*:altion::lation")
    return "\n".join(script_lines)


# Streamlit UI code for uploading files and displaying results
st.title('Abbreviation Suggestion Tool')
st.markdown("""
    This tool helps you generate abbreviations for long words or phrases, making your typing faster and more efficient. 
    Upload your documents, and the tool will analyze the text to suggest useful abbreviations. Use the filters to include your likely abbreviations based on frequency found in your text. 
    You can then download these abbreviations in CSV format, a plist file for [Mac/iOS text replacements](https://support.apple.com/en-gb/guide/mac-help/mchl2a7bd795/mac) or as a [autohotkey](https://www.autohotkey.com) file.
    Want more ideas why abbreviations might be useful? Have a read of [this](https://blog.abreevy8.io/you-dont-have-to-type-faster-to-type-faster/). Bear in mind though the cognitive effort to learn these abbreviations. 
    Use avoid numbers if you don't want numbers in your abbreviations. Often useful for AAC devices. 
    The layout option is really for those who use assistive technology who may have an alternative way of entering text. We are attempting to create quicker shortcuts. Your mileage may vary!
    **NB: We don't save your uploaded documents - we just parse them then display the summarised data here.**
""")
uploaded_files = st.file_uploader("Choose text files", accept_multiple_files=True, type=['txt', 'docx', 'pdf', 'rtf', 'odt'])

if uploaded_files:
    combined_text = read_and_combine_texts(uploaded_files)

if uploaded_files:
    combined_text = read_and_combine_texts(uploaded_files)
    df_all_variants = generate_all_abbreviations(combined_text, layout_mapping, prepend=False)
    df_all_variants_prepend = generate_all_abbreviations(combined_text, layout_mapping, prepend=True)
    prepend_option = st.checkbox("Prepend backslash to abbreviations", value=False)
    
    avoid_numbers_option = st.checkbox("Remove numbers in abbreviations", value=False)
    use_proximity = st.checkbox("Create abbreviations based on letter proximity", value=False)
    layout_option = None
    
    if use_proximity:
        layout_option = st.selectbox(
            "Choose your keyboard/access layout",
            list(layout_mapping.keys())
        )
    
    filter_option = st.selectbox(
        "Select items to display:",
        ('All', 'Just Phrases', 'Just Words'),
        index=0  # Default to showing 'All'
    )
    min_frequency = st.slider("Minimum frequency", min_value=1, max_value=10, value=1)
    
    # Filter the DataFrame based on user selections
    # This is a simplified example; you'll need to adjust it based on your actual DataFrame structure
    if use_proximity and layout_option:
        selected_column = layout_option
    else:
        selected_column = 'Standard' if not avoid_numbers_option else 'Syllable'
    
    # Assuming df_all_variants contains all your abbreviation columns
    if prepend_option:
        # Define the columns to which the prepend logic should be applied
        abbreviation_columns = ['Standard', 'Syllable'] + list(layout_mapping.keys())
        
        # Apply the prepend logic to each relevant column
        for column in abbreviation_columns:
            if column in df_all_variants.columns:  # Check if the column exists in the DataFrame
                df_all_variants[column] = df_all_variants_prepend[column].apply(lambda x: '\\' + x if not x.startswith('\\') else x)
    
    # After updating df_all_variants with the prepend logic, proceed with filtering
    selected_column = 'Standard' if not avoid_numbers_option else 'Syllable'
    if use_proximity and layout_option:
        selected_column = layout_option
    
    # Filter based on user selection
    df_filtered = df_all_variants[['Original', selected_column, 'Frequency']]
    df_filtered = df_filtered.sort_values(by='Frequency', ascending=False)
    df_filtered = df_filtered[df_filtered['Frequency'] >= min_frequency]
    df_filtered = filter_df(df_filtered, filter_option)
    
    # Display the DataFrame without the index
    st.dataframe(df_filtered, width=700, hide_index=True)

    user_typing_speed = st.number_input("Enter your typing speed (WPM)*:", min_value=1, max_value=100, value=40, step=1)
    st.caption("* This can be calculated by you using whatever system you write with and use sentences or words approach at [typefast.io](http://typefast.io)")
    for top_n in [10, 50]:
        total_savings, percentage_increase = calculate_savings(df_filtered, top_n, selected_column, user_typing_speed)
        st.write(f"By learning the top {top_n} abbreviations, you would save {total_savings} keystrokes, "
                 f"leading to an increase in WPM rate by approximately {percentage_increase:.2f}%.")
    
    if df_filtered is not None and not df_filtered.empty:
        # CSV Download
        csv = convert_to_csv(df_filtered,selected_column)
        st.download_button(
            label="⊞  Download abbreviations as CSV",
            data=csv,
            file_name='abbreviations.csv',
            mime='text/csv',
        )
        
        # Text replacements plist
        plist_content = generate_plist_content(df_filtered,selected_column)    
        # Offer the plist for download
        st.download_button(
            label=" Download for Mac/iOS Text Replacements",
            data=plist_content,
            file_name='Text Substitutions.plist',
            mime='application/x-plist'
        )
        ahk_script_content = generate_ahk_script(df_filtered,selected_column)
        st.download_button(
            label="⊞ Download as AutoHotkey Script",
            data=ahk_script_content,
            file_name='abbreviations.ahk',
            mime='text/plain'
        )

    else:
        st.write("No suggestions could be generated.")
