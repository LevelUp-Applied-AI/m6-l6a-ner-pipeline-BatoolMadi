
from ner_pipeline import (
    load_data,
    extract_spacy_entities,
    extract_hf_entities,
    evaluate_ner
)

import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import os

from itertools import combinations
from collections import Counter

import math


# =========================
# Tier 1: Entity counts by category
# =========================
def entity_counts_by_category(entities_df, articles_df):
    merged = entities_df.merge(
        articles_df[['id', 'category']],
        left_on='text_id',
        right_on='id'
    )

    counts = (
        merged
        .groupby(['category', 'entity_label'])
        .size()
        .reset_index(name='count')
    )

    return counts


# =========================
# Tier 1: Heatmap visualization
# =========================
def plot_entity_heatmap(counts_df, save_path="output/heatmap.png"):
    os.makedirs("output", exist_ok=True)

    pivot = counts_df.pivot(
        index='category',
        columns='entity_label',
        values='count'
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")

    plt.title("Entity Distribution by Category")
    plt.xlabel("Entity Label")
    plt.ylabel("Category")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


# =========================
# Tier 1: Evaluation per category
# =========================
def evaluate_per_category(pred_df, gold_df, articles_df):
    results = {}

    gold_merged = gold_df.merge(
        articles_df[['id', 'category']],
        left_on='text_id',
        right_on='id'
    )

    print("Gold categories distribution:")
    print(gold_merged['category'].value_counts())

    categories = articles_df['category'].unique()

    for cat in categories:
        gold_cat = gold_merged[gold_merged['category'] == cat]

        if len(gold_cat) == 0:
            results[cat] = {"precision": None, "recall": None, "f1": None}
            continue

        pred_cat = pred_df[pred_df['text_id'].isin(gold_cat['text_id'])]
        scores = evaluate_ner(pred_cat, gold_cat)

        results[cat] = scores

    return results


# =========================
# Tier 2: Entity Normalization
# =========================
def normalize_entities(entities_df):
    # mapping يدوي
    mapping = {
        "UN": "United Nations",
        "U.N.": "United Nations",
        "IPCC": "IPCC",
        "UNEP": "UNEP"
    }

    def normalize(text):
        return mapping.get(text, text)

    entities_df['entity_text'] = entities_df['entity_text'].apply(normalize)
    return entities_df

# =========================
# Tier 2: Co-occurrence
# =========================
def compute_cooccurrence(entities_df):
    cooccurrence = Counter()

    # group by text
    grouped = entities_df.groupby('text_id')['entity_text'].apply(list)

    for entities in grouped:
        unique_entities = list(set(entities))

        for pair in combinations(unique_entities, 2):
            pair = tuple(sorted(pair))
            cooccurrence[pair] += 1

    return cooccurrence

# =========================
# Tier 2: TF-IDF for entities
# =========================
def compute_entity_tfidf(entities_df):
    # TF
    tf = entities_df['entity_text'].value_counts()

    # DF (in how many docs entity appears)
    df_counts = entities_df.groupby('entity_text')['text_id'].nunique()

    total_docs = entities_df['text_id'].nunique()

    tfidf = {}

    for entity in tf.index:
        tf_val = tf[entity]
        df_val = df_counts[entity]

        idf = math.log(total_docs / (1 + df_val))
        tfidf[entity] = tf_val * idf

    return tfidf

# =========================
# Tier 2: Network Visualization
# =========================
def plot_entity_network(cooccurrence, top_n=20, save_path="output/network.png"):
    import matplotlib.pyplot as plt
    import networkx as nx
    import os

    os.makedirs("output", exist_ok=True)

    # top edges
    top_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:top_n]

    G = nx.Graph()

    for (e1, e2), weight in top_pairs:
        G.add_edge(e1, e2, weight=weight)

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    edges = G.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    weights = [w * 0.5 for w in weights]  # scaling

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=1500,
        font_size=8,
        width=weights
    )

    plt.title("Top Entity Co-occurrence Network")
    plt.savefig(save_path)
    plt.close()


# =========================
# Tier 3: Matching Functions
# =========================
def exact_match(pred, gold):
    return (
        pred['entity_text'] == gold['entity_text'] and
        pred['entity_label'] == gold['entity_label']
    )


def span_overlap(pred, gold):
    return not (
        pred['end_char'] <= gold['start_char'] or
        pred['start_char'] >= gold['end_char']
    )


def partial_match(pred, gold):
    return span_overlap(pred, gold)


def type_agnostic_match(pred, gold):
    return span_overlap(pred, gold)

# =========================
# Tier 3: Core Evaluation
# =========================
def evaluate_custom(pred_df, gold_df, strategy="exact"):
    matched_pred = set()
    matched_gold = set()

    TP = 0

    for i, pred in pred_df.iterrows():
        for j, gold in gold_df.iterrows():

            if strategy == "exact":
                match = exact_match(pred, gold)

            elif strategy == "partial":
                match = partial_match(pred, gold)

            elif strategy == "type":
                match = type_agnostic_match(pred, gold)

            else:
                raise ValueError("Unknown strategy")

            if match and j not in matched_gold:
                TP += 1
                matched_pred.add(i)
                matched_gold.add(j)
                break

    FP = len(pred_df) - len(matched_pred)
    FN = len(gold_df) - len(matched_gold)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

# =========================
# Tier 3: Micro + Macro
# =========================
def evaluate_micro_macro(pred_df, gold_df, strategy="exact"):
    # micro
    micro = evaluate_custom(pred_df, gold_df, strategy)

    # macro
    texts = gold_df['text_id'].unique()
    scores = []

    for tid in texts:
        pred_t = pred_df[pred_df['text_id'] == tid]
        gold_t = gold_df[gold_df['text_id'] == tid]

        score = evaluate_custom(pred_t, gold_t, strategy)
        scores.append(score)

    macro = {
        "precision": sum(s["precision"] for s in scores) / len(scores),
        "recall": sum(s["recall"] for s in scores) / len(scores),
        "f1": sum(s["f1"] for s in scores) / len(scores),
    }

    return {"micro": micro, "macro": macro}

# =========================
# Tier 3: Error Analysis
# =========================
def error_analysis(pred_df, gold_df):
    errors = {
        "boundary_error": 0,
        "type_error": 0,
        "missing_entity": 0,
        "spurious_entity": 0
    }

    matched_gold = set()

    for _, pred in pred_df.iterrows():
        found = False

        for j, gold in gold_df.iterrows():
            if span_overlap(pred, gold):

                if pred['entity_text'] == gold['entity_text']:
                    if pred['entity_label'] != gold['entity_label']:
                        errors["type_error"] += 1
                    else:
                        matched_gold.add(j)

                else:
                    errors["boundary_error"] += 1

                found = True
                break

        if not found:
            errors["spurious_entity"] += 1

    # missing
    errors["missing_entity"] = len(gold_df) - len(matched_gold)

    return errors

# =========================
# Tier 3: Pytest Tests
# =========================
# check tests/test_evaluator.py

if __name__ == "__main__":

    # =========================
    # 🔹 Load Data & Models
    # =========================
    print("\n////////// SETUP //////////")

    df = load_data("data/climate_articles.csv")
    gold_df = pd.read_csv("data/gold_entities.csv")

    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    df_en = df[df['language'] == 'en']

    # =========================
    # 🔹 Extract Entities
    # =========================
    print("\n////////// EXTRACTION //////////")

    spacy_df = extract_spacy_entities(df_en, nlp)
    hf_df = extract_hf_entities(df_en, hf_ner)

    print(f"spaCy entities: {len(spacy_df)}")
    print(f"HF entities: {len(hf_df)}")

    # =========================
    # 🔹 Tier 1
    # =========================
    print("\n////////// Tier 1: Category Analysis //////////")

    # counts
    spacy_counts = entity_counts_by_category(spacy_df, df)
    hf_counts = entity_counts_by_category(hf_df, df)

    # heatmaps
    plot_entity_heatmap(spacy_counts, "output/spacy_heatmap.png")
    plot_entity_heatmap(hf_counts, "output/hf_heatmap.png")

    print("Saved heatmaps to /output")

    # evaluation per category
    spacy_eval = evaluate_per_category(spacy_df, gold_df, df)
    hf_eval = evaluate_per_category(hf_df, gold_df, df)

    print("\nspaCy per-category evaluation:")
    print(spacy_eval)

    print("\nHF per-category evaluation:")
    print(hf_eval)

    # =========================
    # 🔹 Tier 2
    # =========================
    print("\n////////// Tier 2: Entity Aggregation //////////")

    # normalize (copy to avoid overwriting original)
    spacy_norm = normalize_entities(spacy_df.copy())

    # co-occurrence
    cooccurrence = compute_cooccurrence(spacy_norm)

    # tf-idf
    tfidf_scores = compute_entity_tfidf(spacy_norm)

    # optional: filter numeric entities
    tfidf_scores = {
        k: v for k, v in tfidf_scores.items()
        if not any(char.isdigit() for char in k)
    }

    # top entities
    top_entities = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop TF-IDF entities:")
    for e, score in top_entities:
        print(e, round(score, 2))

    # network graph
    plot_entity_network(cooccurrence, save_path="output/entity_network.png")
    print("Saved network graph to /output")

    # =========================
    # 🔹 Tier 3
    # =========================
    print("\n////////// Tier 3: Custom Evaluator //////////")

    # 🔥 IMPORTANT: filter to gold texts only
    gold_ids = gold_df['text_id'].unique()
    spacy_eval_df = spacy_df[spacy_df['text_id'].isin(gold_ids)]

    # strategies
    for strategy in ["exact", "partial", "type"]:
        scores = evaluate_micro_macro(spacy_eval_df, gold_df, strategy)

        print(f"\nStrategy: {strategy}")
        print(scores)

    # error analysis
    errors = error_analysis(spacy_eval_df, gold_df)

    print("\nError Analysis:")
    print(errors)