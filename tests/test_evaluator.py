import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ner_challenges import evaluate_custom

def test_empty_predictions():
    pred = pd.DataFrame(columns=["entity_text", "entity_label", "start_char", "end_char"])
    gold = pd.DataFrame([{
        "entity_text": "Jordan",
        "entity_label": "GPE",
        "start_char": 0,
        "end_char": 6
    }])

    result = evaluate_custom(pred, gold)
    assert result["precision"] == 0


def test_empty_gold():
    pred = pd.DataFrame([{
        "entity_text": "Jordan",
        "entity_label": "GPE",
        "start_char": 0,
        "end_char": 6
    }])
    gold = pd.DataFrame(columns=pred.columns)

    result = evaluate_custom(pred, gold)
    assert result["recall"] == 0


def test_exact_match():
    pred = pd.DataFrame([{
        "entity_text": "Jordan",
        "entity_label": "GPE",
        "start_char": 0,
        "end_char": 6
    }])
    gold = pred.copy()

    result = evaluate_custom(pred, gold)
    assert result["f1"] == 1


def test_partial_overlap():
    pred = pd.DataFrame([{
        "entity_text": "Middle East",
        "entity_label": "LOC",
        "start_char": 0,
        "end_char": 11
    }])
    gold = pd.DataFrame([{
        "entity_text": "East",
        "entity_label": "LOC",
        "start_char": 7,
        "end_char": 11
    }])

    result = evaluate_custom(pred, gold, strategy="partial")
    assert result["recall"] == 1


def test_type_mismatch():
    pred = pd.DataFrame([{
        "entity_text": "Jordan",
        "entity_label": "ORG",
        "start_char": 0,
        "end_char": 6
    }])
    gold = pd.DataFrame([{
        "entity_text": "Jordan",
        "entity_label": "GPE",
        "start_char": 0,
        "end_char": 6
    }])

    result = evaluate_custom(pred, gold)
    assert result["precision"] == 0