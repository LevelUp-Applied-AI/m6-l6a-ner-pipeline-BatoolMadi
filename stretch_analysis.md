# Multilingual NER Analysis

## Arabic vs. English Entity Recognition

The multilingual NER models showed clear differences between English and Arabic entity recognition performance on the climate articles dataset. English texts generally produced more accurate and consistent entity extraction results because English contains capitalization signals and simpler word structures that help models identify named entities more reliably.

Arabic texts were more challenging for both models. Arabic does not use capitalization, has richer morphology, and often contains attached prefixes and suffixes that make entity boundaries harder to detect. In several Arabic climate articles, spaCy’s multilingual model failed to detect any entities, while the Hugging Face XLM-RoBERTa model was able to identify entities more consistently across the Arabic texts. For example, climate-related locations and organizations were detected more reliably by the transformer-based model than by spaCy’s multilingual pipeline.

## Implications for NLP in the MENA Region

These findings highlight the importance of multilingual NLP systems for real-world applications in Jordan and across the MENA region. Climate reports, government policies, and environmental news are commonly written in both English and Arabic, meaning production NLP systems must support bilingual processing instead of relying on English-only pipelines.

The experiment also demonstrates that transformer-based multilingual models can provide stronger Arabic NER performance compared to lighter traditional pipelines. However, Arabic NLP still requires additional domain adaptation and language-specific preprocessing to achieve high-quality entity recognition. Combining multilingual transformer models with custom domain rules could further improve bilingual climate-analysis systems used in professional and governmental environments.