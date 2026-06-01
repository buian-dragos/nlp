import rowordnet
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

# Download NLTK data (only needed once)
nltk.download('wordnet', quiet=True)
nltk.download('sentiwordnet', quiet=True)

print("Loading native RoWordNet... (this takes a few seconds)")
rwn = rowordnet.RoWordNet()

# 3 Nouns, 3 Verbs, 2 Adjectives
target_words = ['câine', 'cafea', 'mașină', 'alerga', 'mânca', 'dormi', 'frumos', 'rapid']

output_file = "rowordnet_results.txt"
print(f"Processing words and saving to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    for word in target_words:
        f.write(f"\n{'=' * 60}\nCUVÂNT: {word.upper()}\n{'=' * 60}\n")

        # Get all synset IDs for the word using native RoWordNet
        synset_ids = rwn.synsets(literal=word)

        if not synset_ids:
            f.write("Nu s-au găsit sensuri în RoWordNet pentru acest cuvânt.\n")
            continue

        for i, synset_id in enumerate(synset_ids, 1):
            # Load the actual concept object
            synset = rwn.synset(synset_id)

            f.write(f"\n--- Sensul {i} ---\n")
            # 1. ID, Literals, Definition (Now in Romanian!)
            f.write(f"ID Synset: {synset.id}\n")
            f.write(f"Literali (Sinonime RO): {', '.join(synset.literals)}\n")
            f.write(f"Definiție: {synset.definition}\n")

            # 2. PNO Scores (The "Bridge" to SentiWordNet)
            try:
                parts = synset.id.split('-')
                if len(parts) == 3:
                    offset = int(parts[1])
                    pos = parts[2]

                    # Ask NLTK to find the matching English concept using the ID number
                    nltk_synset = wn.synset_from_pos_and_offset(pos, offset)
                    senti = swn.senti_synset(nltk_synset.name())

                    f.write(
                        f"Scoruri PNO -> Pozitiv: {senti.pos_score()}, Negativ: {senti.neg_score()}, Obiectiv: {senti.obj_score()}\n")
                else:
                    f.write("Scoruri PNO -> Indisponibile (Format ID invalid).\n")
            except Exception as e:
                f.write("Scoruri PNO -> Indisponibile.\n")

            # 3. Semantic Hierarchy (Hypernyms & Hyponyms)
            outbound_relations = rwn.outbound_relations(synset.id)

            hypernyms = []
            hyponyms = []
            antonyms = []

            # Sort relations into parents (hypernym) and children (hyponym)
            for target_id, relation_type in outbound_relations:
                try:
                    # Look up the actual concept for this related ID
                    related_synset = rwn.synset(target_id)
                    # Grab the primary word (the first literal) from that concept's bucket
                    actual_word = related_synset.literals[0]

                    if relation_type == "hypernym":
                        hypernyms.append(actual_word)
                    elif relation_type == "hyponym":
                        hyponyms.append(actual_word)
                    elif relation_type in ["antonym", "near_antonym"]:
                        antonyms.append(actual_word)
                except Exception:
                    continue

            # Limit hyponyms to 5 to keep the file clean
            hyponyms = hyponyms[:5]

            f.write(f"Hipernime (Mai general): {', '.join(hypernyms) if hypernyms else 'Niciunul'}\n")
            f.write(f"Hiponime (Mai specific): {', '.join(hyponyms) if hyponyms else 'Niciunul'}...\n")
            f.write(f"Antonime (Opus): {', '.join(antonyms) if antonyms else 'Niciunul'}...\n")

print(f"Gata! Native RoWordNet analysis complete. Check {output_file}")