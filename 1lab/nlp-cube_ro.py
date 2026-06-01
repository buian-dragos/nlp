from cube.api import Cube
import sys
import nltk
from nltk.stem.snowball import SnowballStemmer

# 1. Initialize tools
print("Loading NLP Cube model... (this might take a moment)")
try:
    cube = Cube(verbose=True)
    cube.load("ro")
except Exception as e:
    print(f"Error loading Cube: {e}")
    sys.exit(1)

# Initialize NLTK's Romanian Stemmer
stemmer = SnowballStemmer("romanian")

# 10 Romanian Sentences (added 5 more with diverse structures and named entities)
texts = [
    "Salut! Ce mai faci? N-am mai vorbit de mult timp.",
    "Pisica doarme pe canapea. Este foarte liniștită azi.",
    "Mâine mergem la munte. Vremea va fi frumoasă. Abia aștept!",
    "Cafeaua este fierbinte și aromată. Îmi place să o beau dimineața devreme.",
    "Mi-e foame, hai să mâncăm ceva bun. Cunosc un restaurant nou în centru.",
    "Maria a vizitat Parisul anul trecut și a fost impresionată de Turnul Eiffel.",
    "Guvernul României a adoptat o nouă lege privind educația națională.",
    "Câinele vecinului meu latră în fiecare seară la lună.",
    "Am cumpărat două mere, trei pere și o sticlă cu apă minerală.",
    "Dacă vei învăța mai mult, rezultatele tale la examen vor fi excelente."
]

output_file = "romanian_nlpcube_nltk.txt"

# 2. Open the file for writing
with open(output_file, "w", encoding="utf-8") as f:
    for i, text in enumerate(texts, 1):
        doc = cube(text)

        f.write(f"\n{'=' * 90}\n")
        f.write(f"Analiza Textului {i}: {text}\n")
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