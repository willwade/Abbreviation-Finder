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
import string

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
    try:
        text = textract.process(tmp_file_path, encoding='utf-8')
        return text.decode('utf-8')
    except textract.exceptions.ShellError as e:
        print(f"Error processing file {uploaded_file.name}: {e}")
        # Handle the error, e.g., by logging it or returning a default message
        return "Error processing file. Please ensure it's a supported format."



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

def generate_truncated_abbreviation(word_or_phrase):
    """Generate a truncated abbreviation by using the first three letters of every word.
    If a word has fewer than three letters, use the first and last letter."""
    def truncate_word(word):
        if len(word) >= 3:
            return word[:3]
        else:
            # For words with fewer than three letters, use the first and last letter
            # If the word is a single letter, it will simply duplicate that letter
            return word[0] + word[-1]
    return ''.join(truncate_word(word) for word in word_or_phrase.split()).lower()

def generate_contracted_abbreviation(word_or_phrase):
    """Generate a contracted abbreviation by using the first letter, a middle consonant, and the last letter."""
    def contract_word(word):
        if len(word) > 3:
            # Find a middle consonant, if available
            consonants = [char for char in word[1:-1] if char.lower() not in 'aeiou']
            middle = consonants[len(consonants) // 2] if consonants else word[1]
            return f"{word[0]}{middle}{word[-1]}"
        else:
            return word
    return ''.join(contract_word(word) for word in word_or_phrase.split()).lower()

def generate_abbreviation(word_or_phrase, existing_abbreviations, prepend_option=False, no_numbers=False):
    debug = False  # Set to True to enable debug output
    abbreviation_methods = [heuristic_syllables, generate_truncated_abbreviation, generate_contracted_abbreviation]
    abbreviation = ""

    # Ensure existing_abbreviations is a set for efficient lookups
    existing_abbreviations = set(existing_abbreviations)

    # Try each abbreviation method in order
    for method in abbreviation_methods:
        temp_abbr = method(word_or_phrase) if method != heuristic_syllables else ''.join(method(word_or_phrase))
        if debug: print(f"{method.__name__}: {temp_abbr}")
        if temp_abbr not in existing_abbreviations and not is_real_word(temp_abbr):
            abbreviation = temp_abbr
            break  # Found a suitable abbreviation

    # If abbreviation is empty or not unique/real word, modify further
    if not abbreviation or abbreviation in existing_abbreviations or is_real_word(abbreviation):
        abbreviation = word_or_phrase[0]  # Start with the first letter
        if debug: print(f"Initial abbreviation: {abbreviation}")

    original_abbreviation = abbreviation
    counter = 1
    alphabet = string.ascii_lowercase

    while abbreviation in existing_abbreviations or is_real_word(abbreviation) or len(abbreviation) >= len(word_or_phrase):
        if no_numbers:
            # Cycle through the alphabet for additional characters
            next_char = alphabet[(counter - 1) % len(alphabet)]
        else:
            # Append numbers to ensure uniqueness
            next_char = str(counter)
        
        # Attempt to make abbreviation unique and not a real word
        new_abbreviation = original_abbreviation + next_char
        if new_abbreviation not in existing_abbreviations and not is_real_word(new_abbreviation):
            abbreviation = new_abbreviation
            break  # Found a suitable abbreviation
        
        if debug: print(f"Trying abbreviation: {abbreviation}, Counter: {counter}")
        counter += 1
        if counter > 100:  # Prevent infinite loops
            print("Reached counter limit, breaking loop.")
            break

    existing_abbreviations.add(abbreviation)
    return abbreviation


def generate_all_abbreviations(text, layouts, prepend=False):
    # Tokenize the text and filter out stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english') and token.isalpha()]

    # Calculate the frequency of each token
    token_frequency = Counter(filtered_tokens)

    # Find common phrases and calculate their frequencies
    common_phrases_with_freq = find_common_phrases(text, max_length=15, min_frequency=2)
    # Combine token and phrase frequencies
    all_frequencies = {**common_phrases_with_freq, **token_frequency}

    abbreviation_variants = []

    for original, freq in all_frequencies.items():
        if len(original) <= 1 or original in stopwords.words('english'):
            continue

        # Reset existing abbreviations for each type
        existing_standard_abbr = set()
        standard_abbr = generate_abbreviation(original, existing_standard_abbr, prepend)
        standard_abbr_nonum = generate_abbreviation(original, existing_standard_abbr, prepend, True)

        existing_syllable_abbr = set()
        syllable_abbr = generate_syllable_abbreviation(original, existing_syllable_abbr,prepend) if ' ' not in original else standard_abbr

        truncated_abbr = generate_truncated_abbreviation(original)
        contracted_abbr = generate_contracted_abbreviation(original)


        # Generate proximity-based abbreviations for each layout, with separate tracking
        proximity_abbrs = {}
        for layout_name, layout in layouts.items():
            existing_proximity_abbr = set()
            proximity_abbrs[layout_name] = generate_proximity_based_abbreviation(original, layout, existing_proximity_abbr,prepend)

        abbreviation_variants.append({
            'Original': original,
            'Frequency': freq,
            'Standard': standard_abbr,
            'Standard-No Numbers':standard_abbr_nonum,
            'Syllable': syllable_abbr,
            'Truncated': truncated_abbr,
            'Contracted': contracted_abbr,
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

def find_common_phrases(text, max_length=15, min_frequency=2):
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
    common_phrases_with_freq = find_common_phrases(text, max_length=15, min_frequency=2)
    
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

def generate_espanso_yaml_content(df, abbreviation_column):
    yaml_content = "# espanso match file\n\n"
    yaml_content += "# For a complete introduction, visit the official docs at: https://espanso.org/docs/\n\n"
    yaml_content += "# Matches are substitution rules: when you type the \"trigger\" string\n"
    yaml_content += "# it gets replaced by the \"replace\" string.\n"
    yaml_content += "matches:\n"
    
    for _, row in df.iterrows():
        # Skip rows where the abbreviation is None
        if row[abbreviation_column] is None:
            continue
        trigger = row[abbreviation_column].lstrip('\\') if row[abbreviation_column] else ""
        # Skip processing this row if trigger is empty after stripping
        if not trigger:
            continue
        replacement_text = row['Original']
        yaml_content += f"  - trigger: {trigger}\n"
        yaml_content += f"    replace: {replacement_text}\n"
    
    yaml_content += "..."
    
    return yaml_content

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
    Upload your documents, and the tool will analyse the text to suggest valuable abbreviations. Use the filters to include your likely abbreviations based on frequency found in your text. 
    You can then download these abbreviations in a range of formats. 

    Want more ideas why abbreviations might be useful? Have a read of [this](https://blog.abreevy8.io/you-dont-have-to-type-faster-to-type-faster/). Bear in mind though the cognitive effort to learn these abbreviations. 
    There are some options really designed for users who use Assistive Technology to communicate. Your mileage may vary!
    
    What is the easiest abbreviation style to learn? Well its a personal choice but [this paper](https://pubmed.ncbi.nlm.nih.gov/2148561/) found truncation marginally easier. 
    
    **NB: We don't save your uploaded documents - we just parse them then display the summarised data here.**
""")
uploaded_files = st.file_uploader("Choose text files", accept_multiple_files=True, type=['txt', 'docx', 'pdf', 'rtf', 'odt'])

if uploaded_files:
    combined_text = read_and_combine_texts(uploaded_files)
    # Generate abbreviations without prepend logic
    df_all_variants = generate_all_abbreviations(combined_text, layout_mapping, prepend=False)
    df_all_variants_prepend = generate_all_abbreviations(combined_text, layout_mapping, prepend=True)

    abbreviation_options = ["Standard", "Standard-No Numbers", "Syllable", "Truncated", "Contracted", "Positional"]
    selected_abbreviation_strategy = st.selectbox("Select abbreviation strategy:", options=abbreviation_options, index=0)
    st.caption("Standard tries to use Syllable techniques then truncation and then contraction. It tries to minimise having duplicate abbreviations. Other techniques will add numbers to abbreviations if they are dupliicated. Some studies suggest Truncation is easier to learn but note its fixed at 3 letters per word. If you want two-letter abbreviations, use Syllable. Prepend for one letter abbreviations.")
   
    layout_option = None
    if selected_abbreviation_strategy == "Positional":
        layout_option = st.selectbox("Choose your keyboard/access layout", options=list(layout_mapping.keys()))
        st.caption("This is very questionnable particulaly for scanning systems where you generally dont carry on from the last letter selected direct selection for QWERTY etc does make some logical sense")

    prepend_options = {
        "None": "",
        "Colon (:)": ":",
        "Semi-Colon (;)": ";",
        "Backslash (\\)": "\\",
        "Comma (,)": ",",
        "Full stop (.)": "."
    }
    prepend_choice = st.selectbox(
        "Choose a character to prepend to abbreviations:",
        options=list(prepend_options.keys()),
        index=0  # Default to "None"
    )    
    st.caption("Some systems push a user to use a backslah or other character. Some people prefer comma abbreviation as its quicker and unlikely. Just consider how it may work with your abbreviation software ")

    # Get the actual character to prepend based on the user's choice
    prepend_character = prepend_options[prepend_choice]
    

    filter_option = st.selectbox("Select items to display:", ('All', 'Just Phrases', 'Just Words'), index=0)
    min_frequency = st.slider("Minimum frequency", min_value=1, max_value=10, value=1)
    min_word_length = st.selectbox(
        "Only consider words of at least this length:",
        options=[0, 3, 4, 5],
        index=1,  # Default to ignoring words less than 3 letters long
        format_func=lambda x: f"{x} letters" if x > 0 else "No filter"
    )

    
    # Determine the selected column based on the strategy and layout option
    selected_column = selected_abbreviation_strategy if selected_abbreviation_strategy != "Positional" else layout_option
    
    # Filter and sort the DataFrame based on user selections
    if prepend_character:
        df_filtered = df_all_variants_prepend[['Original', selected_column, 'Frequency']]
    else:
        df_filtered = df_all_variants[['Original', selected_column, 'Frequency']]
        
    df_filtered = df_filtered[df_filtered['Frequency'] >= min_frequency]
    df_filtered = filter_df(df_filtered, filter_option).sort_values(by='Frequency', ascending=False)
    if min_word_length > 0:
        df_filtered = df_filtered[df_filtered['Original'].apply(lambda x: len(x) >= min_word_length or ' ' in x)]

    # Apply the prepend logic to the selected column if the option is selected
    if prepend_character:
        df_filtered[selected_column] = df_filtered[selected_column].apply(lambda x: prepend_character + x if x and not x.startswith(prepend_character) else x)

    # Display the DataFrame without the index
    st.dataframe(df_filtered, width=700, hide_index=True)

    user_typing_speed = st.number_input("Enter your typing speed (WPM)*:", min_value=1, max_value=100, value=40, step=1)
    st.caption("This can be calculated by you using whatever system you write with and use sentences or words approach at [typefast.io](http://typefast.io)")
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
        st.caption("[See this guide](https://support.apple.com/en-gb/guide/mac-help/mchl2a7bd795/mac) on how to use for MacOS ")

        ahk_script_content = generate_ahk_script(df_filtered,selected_column)
        st.download_button(
            label="⊞ Download as AutoHotkey Script",
            data=ahk_script_content,
            file_name='abbreviations.ahk',
            mime='text/plain'
        )
        st.caption("Use this with [AutoHotKey on Windows](https://www.autohotkey.com) on its own or as part of a bigger tool  e.g this [AutoCorrectTool](https://www.autohotkey.com/boards/viewtopic.php?f=83&t=120220&start=60#p565896)")
        
        yaml_content = generate_espanso_yaml_content(df_filtered,selected_column) 
        st.download_button(
            label="Download for Espanso",
            data=yaml_content,
            file_name='personal_abbreviations.yml',
            mime='text/yaml'
        )
        st.caption("[Espanso](https://espanso.org) is a free and opensource tool for abbreviation expansion and much more. Its cross platform. Check it out. ")


    else:
        st.write("No suggestions could be generated.")
