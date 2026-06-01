import spacy
from spacy import displacy
import nltk
from nltk.stem import PorterStemmer

# Initialize NLTK Stemmer
stemmer = PorterStemmer()

# Load the English model
print("Loading SpaCy English model...")
nlp = spacy.load("en_core_web_sm")

# 10 English texts
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

output_file = "spacy_eng.txt"
print(f"Analyzing texts and writing output to {output_file}...")

# Open the file for writing
with open(output_file, "w", encoding="utf-8") as f:
    # Process and display texts with all operations
    for i, text in enumerate(texts, 1):
        doc = nlp(text)
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Text {i} Analysis: {text}\n")
        f.write(f"{'=' * 80}\n")

        # 1. Tokenization, Stemming, Lemmatization, POS (General & Detailed), Dependencies
        f.write(f"{'Token':<15} {'Stem':<15} {'Lemma':<15} {'POS':<6} {'TAG':<6} {'Dep':<10} {'Head'}\n")
        f.write('-' * 80 + '\n')
        for token in doc:
            stemmed_word = stemmer.stem(token.text)
            f.write(f"{token.text:<15} {stemmed_word:<15} {token.lemma_:<15} {token.pos_:<6} {token.tag_:<6} {token.dep_:<10} {token.head.text}\n")

        # 2. Chunking (Noun Chunks)
        f.write("\n--- Noun Chunks ---\n")
        chunks = list(doc.noun_chunks)
        if chunks:
            for chunk in chunks:
                f.write(f"- {chunk.text} (Root: {chunk.root.text}, Dep: {chunk.root.dep_})\n")
        else:
            f.write("No noun chunks found.\n")

        # 3. NER (Named Entity Recognition)
        f.write("\n--- Named Entities (NER) ---\n")
        if doc.ents:
            for ent in doc.ents:
                f.write(f"- {ent.text} ({ent.label_})\n")
        else:
            f.write("No named entities found.\n")

print(f"Success! Analysis written to {output_file}")

# 4. Visualize dependency graphs for ALL texts
print("\nVisualizing dependency graphs for ALL texts.")
print("Check your browser at http://localhost:5000")
print("-> Use the 'Next' and 'Previous' arrows at the bottom of the webpage to browse through the sentences!")
print("Press Ctrl+C in this terminal to stop the server when you are done.")

# Parse all texts into a list of Doc objects and pass the whole list to displacy
docs = list(nlp.pipe(texts))
displacy.serve(docs, style='dep', options={'distance': 100})