from cube.api import Cube
import sys
import nltk
from nltk.stem.snowball import SnowballStemmer

# 1. Initialize tools
print("Loading NLP Cube English model... (this might take a moment)")
try:
    cube = Cube(verbose=True)
    # Load the English model instead of Romanian
    cube.load("en")
except Exception as e:
    print(f"Error loading Cube: {e}")
    sys.exit(1)

# Initialize NLTK's English Stemmer
stemmer = SnowballStemmer("english")

# 10 English Sentences
texts = [
    "Hey there! How's it going? I'm excited for the weekend.",
    "The cat chased the mouse. It hid under the sofa.",
    "Spacy is amazing for NLP tasks. It simplifies things a lot.",
    "We visited New York last summer. It was crowded but fun.",
    "Coffee wakes me up. Tea relaxes me in the evenings.",
    "Apple Inc. announced a new iPhone on Tuesday in California.",
    "The quick brown fox jumps over the lazy dog.",
    "Elon Musk acquired Twitter for 44 billion dollars in 2022.",
    "If it rains tomorrow, the committee will cancel the picnic.",
    "Mount Everest is the highest mountain on Earth."
]

output_file = "english_nlpcube_nltk.txt"

# 2. Open the file for writing
with open(output_file, "w", encoding="utf-8") as f:
    for i, text in enumerate(texts, 1):
        doc = cube(text)

        f.write(f"\n{'=' * 90}\n")
        f.write(f"Text {i} Analysis: {text}\n")
        f.write(f"{'=' * 90}\n")

        # Headers for all required operations
        header_cols = f"{'Token':<15} {'Stem':<15} {'Lemma':<15} {'UPOS':<6} {'XPOS':<6} {'Dep':<10} {'Head'} {'Morphology (Attrs)'}\n"
        f.write(header_cols)
        f.write("-" * 90 + "\n")

        # Iterate through sentences and tokens
        for sentence in doc.sentences:
            for token in sentence.words:
                # NLTK Stemming
                stemmed_word = stemmer.stem(token.word)

                # Format the line with all attributes
                # token.word  = Tokenization
                # token.lemma = Lemmatization
                # token.upos  = General POS tagging (Universal POS)
                # token.xpos  = Detailed POS tagging (Language-specific)
                # token.label = Dependency relation
                # token.head  = Dependency parsing (Head ID)
                # token.attrs = Morphological features

                line = f"{token.word:<15} {stemmed_word:<15} {token.lemma:<15} {token.upos:<6} {token.xpos:<6} {token.label:<10} {token.head:<4} {token.attrs}\n"
                f.write(line)

    print(f"\nSuccess! Detailed analysis written to: {output_file}")