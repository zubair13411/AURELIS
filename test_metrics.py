import pytest
from src.metrics import calculate_metrics


def test_confusion_counts_and_rates():
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert (metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']) == (1, 1, 1, 1)
    assert metrics['accuracy'] == 0.5
    assert metrics['specificity'] == 0.5
    assert metrics['recall_pneumonia'] == 0.5


def test_always_pneumonia_baseline_has_zero_specificity():
    metrics = calculate_metrics([0, 0, 1, 1], [1, 1, 1, 1])
    assert metrics['fp'] == 2
    assert metrics['fn'] == 0
    assert metrics['specificity'] == 0.0


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_metrics([], [])
