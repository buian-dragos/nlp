import spacy
from spacy import displacy
import nltk
from nltk.stem.snowball import SnowballStemmer

# Initialize NLTK Stemmer for Romanian
stemmer = SnowballStemmer("romanian")

# Load the Romanian model
print("Loading SpaCy Romanian model...")
nlp = spacy.load("ro_core_news_sm")

# The 10 Romanian sentences
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

output_file = "spacy_ro.txt"
print(f"Analyzing texts and writing output to {output_file}...")

# Open the file for writing
with open(output_file, "w", encoding="utf-8") as f:
    # Process and display texts with all operations
    for i, text in enumerate(texts, 1):
        doc = nlp(text)
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Analiza Textului {i}: {text}\n")
        f.write(f"{'=' * 80}\n")

        # 1. Tokenization, Stemming, Lemmatization, POS (General & Detailed), Dependencies
        f.write(f"{'Token':<15} {'Stem':<15} {'Lemma':<15} {'POS':<6} {'TAG':<6} {'Dep':<10} {'Head'}\n")
        f.write('-' * 80 + '\n')
        for token in doc:
            stemmed_word = stemmer.stem(token.text)
            f.write(
                f"{token.text:<15} {stemmed_word:<15} {token.lemma_:<15} {token.pos_:<6} {token.tag_:<6} {token.dep_:<10} {token.head.text}\n")

        # 2. Chunking (Grupuri Nominale - CUSTOM ROMANIAN CHUNKER)
        f.write("\n--- Noun Chunks (Grupuri Nominale) ---\n")
        chunks = []
        for token in doc:
            # Look for Nouns and Proper Nouns
            if token.pos_ in ["NOUN", "PROPN"]:
                # Grab the noun's determiners, adjectives, and numbers
                modifiers = [child for child in token.children if child.dep_ in ["det", "amod", "nummod"]]
                # Combine them with the root noun
                chunk_tokens = modifiers + [token]
                # Sort them so they are in the correct original sentence order
                chunk_tokens.sort(key=lambda t: t.i)

                # Join the words together into a single string
                chunk_text = " ".join([t.text for t in chunk_tokens])
                chunks.append((chunk_text, token.text, token.dep_))

        # Write the chunks to the file
        if chunks:
            for chunk_text, root_text, root_dep in chunks:
                f.write(f"- {chunk_text} (Root: {root_text}, Dep: {root_dep})\n")
        else:
            f.write("Nu s-au găsit grupuri nominale.\n")

        # 3. NER (Named Entity Recognition)
        f.write("\n--- Named Entities (Entități Numite) ---\n")
        if doc.ents:
            for ent in doc.ents:
                f.write(f"- {ent.text} ({ent.label_})\n")
        else:
            f.write("Nu s-au găsit entități numite.\n")

print(f"Success! Analysis written to {output_file}")

# 4. Visualize dependency graphs for ALL texts
print("\nVizualizare grafic de dependență pentru TOATE textele.")
print("Verifică browserul la http://localhost:5000")
print("-> Folosește săgețile 'Next' și 'Previous' din partea de jos pentru a naviga prin propoziții!")
print("Apasă Ctrl+C în acest terminal pentru a opri serverul când ai terminat.")

# Parse all texts into a list of Doc objects and pass the whole list to displacy
docs = list(nlp.pipe(texts))
displacy.serve(docs, style='dep', options={'distance': 100})