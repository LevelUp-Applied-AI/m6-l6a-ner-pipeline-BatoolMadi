"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline
import unicodedata

def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame
    return pd.read_csv(filepath)


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # TODO: Compute shape, language/category value_counts, and word-count
    #       statistics on df['text']
    
    # shape
    shape = df.shape

    # language distribution
    lang_counts = df['language'].value_counts().to_dict()

    # category distribution
    category_counts = df['category'].value_counts().to_dict()

    # text length stats (word count)
    word_counts = df['text'].astype(str).apply(lambda x: len(x.split()))

    text_length_stats = {
        "mean": word_counts.mean(),
        "min": word_counts.min(),
        "max": word_counts.max()
    }

    return {
        "shape": shape,
        "lang_counts": lang_counts,
        "category_counts": category_counts,
        "text_length_stats": text_length_stats
    }

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # TODO: NFC-normalize the text, run it through nlp(), drop
    #       punctuation/whitespace tokens, return lowercased lemmas
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Lemmatization (English only)
    doc = nlp(text)

    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

    return tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    results = []

    for _, row in df.iterrows():
        text = row['text']
        text_id = row['id']

        doc = nlp(text)

        for ent in doc.ents:
            results.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(results)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, run each text through
    #       ner_pipeline, merge ## subword tokens, strip B-/I- prefix
    #       from labels (IOB format), return as a DataFrame
    results = []

    # 1. Filter English only
    df_en = df[df['language'] == 'en']

    for _, row in df_en.iterrows():
        text = row['text']
        text_id = row['id']

        # 2. Run HF pipeline
        raw_entities = ner_pipeline(text)

        merged_entities = []
        current = None

        for ent in raw_entities:
            word = ent['word']
            label = ent['entity']
            start = ent['start']
            end = ent['end']

            # 3. Merge subwords (##)
            if word.startswith("##") and current is not None:
                current['entity_text'] += word[2:]
                current['end_char'] = end
            else:
                # save previous entity
                if current is not None:
                    merged_entities.append(current)

                # remove B- / I- prefix
                clean_label = label.split("-")[-1]

                current = {
                    "text_id": text_id,
                    "entity_text": word,
                    "entity_label": clean_label,
                    "start_char": start,
                    "end_char": end
                }

        # append last entity
        if current is not None:
            merged_entities.append(current)

        results.extend(merged_entities)

    # 4. Return DataFrame
    return pd.DataFrame(results)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    # TODO: Count entities per label for each system, compute totals,
    #       and derive the three overlap sets by matching on
    #       (text_id, entity_text)
    #  total counts
    total_spacy = len(spacy_df)
    total_hf = len(hf_df)

    #  counts by label
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict()
    hf_counts = hf_df['entity_label'].value_counts().to_dict()

    #  create sets for comparison (text_id, entity_text)
    spacy_set = set(
        zip(spacy_df['text_id'], spacy_df['entity_text'])
    )
    hf_set = set(
        zip(hf_df['text_id'], hf_df['entity_text'])
    )

    #  overlaps
    both = spacy_set & hf_set
    spacy_only = spacy_set - hf_set
    hf_only = hf_set - spacy_set

    #  print summary (مهم للعرض)
    print("=== NER Comparison Summary ===")
    print(f"Total spaCy entities: {total_spacy}")
    print(f"Total HF entities: {total_hf}")
    print()

    print("spaCy label counts:")
    print(spacy_counts)
    print()

    print("HF label counts:")
    print(hf_counts)
    print()

    print(f"Entities found by BOTH: {len(both)}")
    print(f"spaCy ONLY: {len(spacy_only)}")
    print(f"HF ONLY: {len(hf_only)}")

    return {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": total_spacy,
        "total_hf": total_hf,
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Important: filter `predicted_df` to the set of text_ids in
    `gold_df` before counting. Predictions on texts that have no gold
    annotations cannot be true positives — counting them as false
    positives produces misleading order-of-magnitude precision drops
    on a sparse gold standard.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # TODO: Filter predicted_df to gold_df['text_id'].unique();
    #       match remaining predictions to gold entities by text_id +
    #       entity_text + entity_label, compute precision/recall/F1
    
    #  1. فلترة predicted على نفس text_ids الموجودة بالـ gold
    valid_ids = set(gold_df['text_id'])
    predicted_df = predicted_df[predicted_df['text_id'].isin(valid_ids)]

    #  2. نحول لـ sets للمقارنة (text_id, entity_text, entity_label)
    pred_set = set(
        zip(predicted_df['text_id'],
            predicted_df['entity_text'],
            predicted_df['entity_label'])
    )

    gold_set = set(
        zip(gold_df['text_id'],
            gold_df['entity_text'],
            gold_df['entity_label'])
    )

    #  3. نحسب TP / FP / FN
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    #  4. نحسب Precision / Recall / F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def extract_multilingual_entities(df, multilingual_nlp):
    """Extract named entities from Arabic-language texts using a
    multilingual spaCy model.

    Run the *English* spaCy model on Arabic text and you will see
    garbage output — `en_core_web_sm` is English-only and produces
    spurious or empty results on non-English scripts. The fix is a
    multilingual model (`xx_ent_wiki_sm`) that was trained on the
    multilingual Wikipedia entity dataset and handles Arabic, Chinese,
    French, and many other languages.

    Args:
        df: DataFrame with columns id, text, language, ...
        multilingual_nlp: A loaded spaCy multilingual Language object
            (e.g., `xx_ent_wiki_sm`).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to Arabic rows (language == 'ar'), process each
    #       text with multilingual_nlp, collect entities into rows,
    #       return as a DataFrame. Use the same column structure as
    #       extract_spacy_entities and extract_hf_entities.

    results = []

    #  1. فلترة العربي فقط
    df_ar = df[df['language'] == 'ar']

    for _, row in df_ar.iterrows():
        text = row['text']
        text_id = row['id']

        doc = multilingual_nlp(text)

        for ent in doc.ents:
            results.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")
    multilingual_nlp = spacy.load("xx_ent_wiki_sm")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")

        # Task 7: Arabic NER with multilingual model
        arabic_entities = extract_multilingual_entities(df, multilingual_nlp)
        if arabic_entities is not None:
            print(f"\nArabic entities (multilingual model): {len(arabic_entities)} total")
