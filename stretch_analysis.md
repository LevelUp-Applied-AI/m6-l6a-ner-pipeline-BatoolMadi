## Custom NER Rules Analysis

The custom EntityRuler improved the detection of domain-specific climate entities that were not consistently recognized by the base spaCy model, such as "COP28", "Paris Agreement", and "Sixth Assessment Report".

When applied **before the NER component**, the ruler took priority and injected high-confidence matches into the pipeline. This resulted in a measurable improvement in precision (from 0.6567 to 0.6984) and F1 score (from 0.6518 to 0.6717), while recall remained unchanged. This suggests that the rules helped reduce ambiguity and improved the correctness of extracted entities without significantly increasing coverage.

When applied **after the NER component**, the results were identical to the baseline. This indicates that the statistical NER model had already assigned entity spans, and the ruler did not override them. Therefore, its impact was minimal in this configuration.

Qualitatively, the rules successfully captured domain-specific terms such as "Sixth Assessment Report", which are critical in climate-related texts. However, some patterns (e.g., threshold patterns like percentages or temperature values) may introduce noise if applied too broadly.

Overall, the experiment demonstrates that rule-based systems can significantly enhance NER performance when applied before statistical models, but their effectiveness depends heavily on pattern specificity and pipeline placement.