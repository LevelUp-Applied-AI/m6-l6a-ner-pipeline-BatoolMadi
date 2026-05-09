import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def compute_embeddings(texts, tokenizer, model):

    embeddings = []

    model.eval()

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)

        embeddings.append(
            embedding.squeeze().numpy()
        )

    return np.array(embeddings)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/climate_articles.csv")

df = df.dropna(subset=["text"])

df["language"] = (
    df["language"]
    .astype(str)
    .str.strip()
    .str.lower()
)

english_df = df[df["language"] == "en"].head(10)

arabic_df = df[df["language"] == "ar"].head(10)

english_texts = english_df["text"].tolist()

arabic_texts = arabic_df["text"].tolist()

all_texts = english_texts + arabic_texts

print(f"English texts: {len(english_texts)}")
print(f"Arabic texts: {len(arabic_texts)}")


# =========================
# LOAD MULTILINGUAL BERT
# =========================

model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)

print("\nModel loaded.")


# =========================
# COMPUTE EMBEDDINGS
# =========================

embeddings = compute_embeddings(
    all_texts,
    tokenizer,
    model
)

print(f"\nEmbeddings shape: {embeddings.shape}")


# =========================
# SIMILARITY MATRIX
# =========================

similarity_matrix = cosine_similarity(
    embeddings
)

print("\nSimilarity matrix shape:")
print(similarity_matrix.shape)


# =========================
# LABELS
# =========================

labels = []

for text in english_texts:

    labels.append(
        "EN: " + text[:40].replace("\n", " ")
    )

for text in arabic_texts:

    labels.append(
        "AR: " + text[:40].replace("\n", " ")
    )


# =========================
# HEATMAP
# =========================

plt.figure(figsize=(14, 12))

sns.heatmap(
    similarity_matrix,
    xticklabels=labels,
    yticklabels=labels,
    cmap="viridis"
)

plt.title(
    "Cross-Lingual Embedding Similarity Heatmap"
)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "cross_lingual_heatmap.png",
    dpi=300
)

print("\nHeatmap saved as cross_lingual_heatmap.png")


# =========================
# ANALYSIS HELPERS
# =========================

print("\nTop Cross-Lingual Similarities:\n")

for i in range(10):

    for j in range(10, 20):

        score = similarity_matrix[i][j]

        print(
            f"EN {i} <-> AR {j-10} "
            f"Similarity: {score:.4f}"
        )