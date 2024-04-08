import re
import nltk
from nltk.corpus import words

nltk.download('words')  # Ensure the 'words' corpus is available
english_words = set(words.words())  # This should now work without errors

def is_real_word(abbreviation):
    return abbreviation.lower() in english_words

def heuristic_syllables(word):
    syllables = re.findall(r'[bcdfghjklmnpqrstvwxyz]*[aeiouy]+', word.lower())
    return [syl[0] for syl in syllables]

def generate_abbreviation(word_or_phrase, existing_abbreviations, prepend_option=False):
    if ' ' in word_or_phrase:  # It's a phrase
        abbreviation = ''.join(word[0] for word in word_or_phrase.split()).lower()
    else:  # It's a single word
        syllable_initials = heuristic_syllables(word_or_phrase)
        abbreviation = ''.join(syllable_initials)

    # Prepend backslash if option is selected
    if prepend_option:
        abbreviation = '\\' + abbreviation

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

existing_standard_abbr = set()
abbrevs = generate_abbreviation('maths', existing_standard_abbr, False)
print(abbrevs)
