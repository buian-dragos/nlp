import spacy
from gensim.models import KeyedVectors
import gc  # Garbage collector to free up RAM between models

print("Loading SpaCy Romanian model...")
nlp = spacy.load("ro_core_news_sm")

# Map exactly to the files you downloaded
model_files = {
    "1_WORD": "corola.300.20.vec",
    "2_LEMMA": "lemma_corola.300.50.vec",
    "3_LEMMA+POS": "pos_corola.300.50.vec"
}

output_file = "corola_final_results.txt"

# Your target words
input_words = ["regele", "femeie", "câine", "pisică", "mașină", "alerg", "mănâncă", "dormi", "frumos", "inteligent"]

# Your analogies
analogies_to_test = [
    ("Gen (Masculin -> Feminin)", ["rege", "femeie"], ["bărbat"]),
    ("Geografie (Capitală -> Țară)", ["bucurești", "franța"], ["românia"]),
]


# Helper function to transform words based on the active model
def transform_word(word, mode):
    doc = nlp(word)
    lemma = doc[0].lemma_
    pos = doc[0].pos_

    if mode == "1_WORD":
        return word
    elif mode == "2_LEMMA":
        return lemma
    elif mode == "3_LEMMA+POS":
        return f"{lemma}_{pos}"


print(f"Processing words and saving to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== ANALIZA COROLA (Cele 3 Modele Separate) ===\n")

    # Loop through each of your 3 .vec files
    for mode, file_path in model_files.items():
        print(f"\n[{mode}] Loading model from '{file_path}'...")
        f.write(f"\n\n{'=' * 70}\nMODEL: {mode} ({file_path})\n{'=' * 70}\n")

        try:
            # Load the model into RAM
            model = KeyedVectors.load_word2vec_format(file_path, binary=False)
        except Exception as e:
            f.write(f"Eroare la încărcare: {e}\n")
            print(f"CRASH: Failed to load {file_path}")
            continue

        # --- NEAREST NEIGHBORS ---
        f.write("\n--- CEI MAI Apropiați 10 VECINI ---\n")
        for word in input_words:
            # Transform the word to match the current model
            query_string = transform_word(word, mode)

            f.write(f"\nCuvânt Original: '{word}' -> Interogare AI: '{query_string}'\n")
            try:
                neighbors = model.most_similar(query_string, topn=10)
                for w, score in neighbors:
                    f.write(f"   - {w}: {score:.4f}\n")
            except KeyError:
                f.write("   [!] Nu există în vocabularul acestui model.\n")

        # --- ANALOGIES ---
        f.write("\n\n--- ANALOGII SEMANTICE ---\n")
        for test_name, pos_words, neg_words in analogies_to_test:

            # Transform all the words in the analogy to match the model (e.g., word -> lemma)
            trans_pos = [transform_word(w, mode) for w in pos_words]
            trans_neg = [transform_word(w, mode) for w in neg_words]

            equation = f"{' + '.join(trans_pos)} - {' - '.join(trans_neg)}"
            f.write(f"\nTest: {test_name}\nEcuație: {equation} = \n")

            try:
                results = model.most_similar(positive=trans_pos, negative=trans_neg, topn=3)
                for w, score in results:
                    f.write(f"  -> {w} (Scor: {score:.4f})\n")
            except KeyError as e:
                f.write(f"  [!] Eroare: Cuvântul {e} lipsește din vocabularul acestui model.\n")

        # VERY IMPORTANT: Delete the model from RAM before loading the next one!
        print(f"[{mode}] Unloading model to free up memory...")
        del model
        gc.collect()

print(f"\nDone! All 3 models processed successfully. Check {output_file}")