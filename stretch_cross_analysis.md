# Cross-Lingual Embedding Analysis

## Cross-Lingual Similarity Results

The multilingual BERT model (`bert-base-multilingual-cased`) successfully captured semantic similarity between English and Arabic climate-related texts. Most English-Arabic similarity scores ranged between approximately `0.55` and `0.75`, which indicates that semantically related documents were positioned relatively close together in the shared embedding space despite being written in different languages.

Several text pairs achieved particularly high similarity scores. For example:

* EN 0 ↔ AR 2 → `0.7516`
* EN 7 ↔ AR 3 → `0.7327`
* EN 0 ↔ AR 0 → `0.7375`

These results suggest that the multilingual model learned cross-lingual semantic representations rather than relying only on exact word overlap. Lower similarity scores were generally observed between texts discussing unrelated climate topics, showing that the model still preserved topical separation across languages.

---

## Implications for Bilingual NLP in the MENA Region

These findings suggest that multilingual transformer embeddings can support bilingual NLP systems in the MENA region without requiring separate embedding models for Arabic and English. A single multilingual embedding space could power applications such as semantic search, retrieval systems, document clustering, recommendation systems, and retrieval-augmented generation (RAG) pipelines across both languages.

However, the similarity scores were still lower than what is typically observed between same-language documents, indicating that multilingual models may sacrifice some language-specific precision for broader multilingual capability. Additionally, Arabic tokenization and domain-specific terminology may reduce embedding quality for specialized climate content. In production systems, this could be improved using larger multilingual models, domain-specific fine-tuning, or bilingual retrieval pipelines optimized for Arabic-English climate data.
