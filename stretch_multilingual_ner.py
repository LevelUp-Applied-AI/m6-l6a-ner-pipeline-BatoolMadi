import pandas as pd
import spacy
from transformers import pipeline


# Load dataset
df = pd.read_csv("data/climate_articles.csv")

# Select 20 English and 20 Arabic texts
english_df = df[df["language"] == "en"].head(20)
arabic_df = df[df["language"] == "ar"].head(20)

# Load multilingual spaCy model
spacy_nlp = spacy.load("xx_ent_wiki_sm")

# Load Hugging Face multilingual model
hf_ner = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-wikiann-ner",
    aggregation_strategy="simple"
)

results = []


def process_spacy(texts, language):

    entity_counts = {}
    examples = []

    total_entities = 0
    total_words = 0
    no_entity_count = 0

    for text in texts:

        doc = spacy_nlp(text)

        ents = doc.ents

        total_words += len(text.split())

        if len(ents) == 0:
            no_entity_count += 1

        total_entities += len(ents)

        for ent in ents:

            entity_counts[ent.label_] = (
                entity_counts.get(ent.label_, 0) + 1
            )

            if len(examples) < 3:
                examples.append(
                    f"{ent.text} ({ent.label_})"
                )

    density = (
        total_entities / total_words
    ) * 100

    results.append({
        "model": "spaCy xx_ent_wiki_sm",
        "language": language,
        "total_entities": total_entities,
        "entity_density_per_100_words": round(density, 2),
        "entity_types": str(entity_counts),
        "example_entities": str(examples),
        "no_entity_texts": no_entity_count
    })


def process_huggingface(texts, language):

    entity_counts = {}
    examples = []

    total_entities = 0
    total_words = 0
    no_entity_count = 0

    for text in texts:

        ents = hf_ner(text)

        total_words += len(text.split())

        if len(ents) == 0:
            no_entity_count += 1

        total_entities += len(ents)

        for ent in ents:

            label = ent["entity_group"]

            entity_counts[label] = (
                entity_counts.get(label, 0) + 1
            )

            if len(examples) < 3:
                examples.append(
                    f"{ent['word']} ({label})"
                )

    density = (
        total_entities / total_words
    ) * 100

    results.append({
        "model": "HF XLM-RoBERTa",
        "language": language,
        "total_entities": total_entities,
        "entity_density_per_100_words": round(density, 2),
        "entity_types": str(entity_counts),
        "example_entities": str(examples),
        "no_entity_texts": no_entity_count
    })


# Run analyses
process_spacy(
    english_df["text"],
    "English"
)

process_spacy(
    arabic_df["text"],
    "Arabic"
)

process_huggingface(
    english_df["text"],
    "English"
)

process_huggingface(
    arabic_df["text"],
    "Arabic"
)

# Create comparison table
comparison_df = pd.DataFrame(results)

print("\nMultilingual NER Comparison")
print(comparison_df)

# Save CSV
comparison_df.to_csv(
    "comparison_table.csv",
    index=False
)