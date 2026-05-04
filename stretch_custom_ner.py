import spacy
import pandas as pd
from spacy.pipeline import EntityRuler

from ner_pipeline import load_data, extract_spacy_entities, evaluate_ner

# =========================
# 1. Define Custom Patterns
# =========================

def build_entity_ruler(ruler):
    patterns = [
        {"label": "AGREEMENT", "pattern": "Paris Agreement"},
        {"label": "AGREEMENT", "pattern": "Kyoto Protocol"},

        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},

        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},

        {"label": "POLICY", "pattern": "net zero emissions"},
        {"label": "POLICY", "pattern": "carbon neutrality"},

        {"label": "THRESHOLD", "pattern": [{"LIKE_NUM": True}, {"TEXT": "°C"}]},
        {"label": "THRESHOLD", "pattern": [{"LIKE_NUM": True}, {"TEXT": "%"}]},
    ]

    ruler.add_patterns(patterns)


# =========================
# 2. Apply NER Pipeline
# =========================

def extract_entities_with_nlp(df, nlp):
    records = []

    for _, row in df.iterrows():
        doc = nlp(row["text"])

        for ent in doc.ents:
            records.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(records)


# =========================
# 3. Filter Standard Labels
# =========================

STANDARD_LABELS = {
    "ORG", "GPE", "DATE", "LAW", "MONEY",
    "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"
}

def filter_standard_labels(df):
    return df[df["entity_label"].isin(STANDARD_LABELS)]


# =========================
# 4. Main Experiment
# =========================

if __name__ == "__main__":

    print("////////// LOAD DATA //////////")
    df = load_data("data/climate_articles.csv")
    gold_df = pd.read_csv("data/gold_entities.csv")

    df_en = df[df["language"] == "en"]

    # =========================
    # BASELINE (no rules)
    # =========================
    print("\n////////// BASELINE NER //////////")
    nlp_base = spacy.load("en_core_web_sm")

    base_entities = extract_entities_with_nlp(df_en, nlp_base)
    base_eval = evaluate_ner(
        filter_standard_labels(base_entities),
        gold_df
    )

    print("Baseline evaluation:", base_eval)

    # =========================
    # RULER BEFORE NER
    # =========================
    print("\n////////// RULER BEFORE NER //////////")
    nlp_before = spacy.load("en_core_web_sm")
    ruler_before = nlp_before.add_pipe("entity_ruler", before="ner")
    build_entity_ruler(ruler_before)

    before_entities = extract_entities_with_nlp(df_en, nlp_before)
    before_eval = evaluate_ner(
        filter_standard_labels(before_entities),
        gold_df
    )

    print("Before NER evaluation:", before_eval)

    # =========================
    # RULER AFTER NER
    # =========================
    print("\n////////// RULER AFTER NER //////////")
    nlp_after = spacy.load("en_core_web_sm")
    ruler_after = nlp_after.add_pipe("entity_ruler", after="ner")
    build_entity_ruler(ruler_after)

    after_entities = extract_entities_with_nlp(df_en, nlp_after)
    after_eval = evaluate_ner(
        filter_standard_labels(after_entities),
        gold_df
    )

    print("After NER evaluation:", after_eval)

    # =========================
    # Comparison
    # =========================
    print("\n////////// COMPARISON //////////")
    print("Baseline:", base_eval)
    print("Before:", before_eval)
    print("After:", after_eval)

    # =========================
    # Sample Outputs
    # =========================
    print("\n////////// SAMPLE MATCHES //////////")
    sample_text = df_en.iloc[0]["text"]

    print("\nText:", sample_text)
    print("\nEntities AFTER rules:")

    doc = nlp_after(sample_text)
    for ent in doc.ents:
        print(ent.text, "->", ent.label_)