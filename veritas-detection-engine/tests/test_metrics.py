import math

from src.eval.metrics import binary_metrics, expected_calibration_error, roc_auc


def test_binary_metrics_known_case():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5


def test_auc_known_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]) == 0.75
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert math.isnan(roc_auc([1, 1], [0.2, 0.9]))


def test_perfect_calibration_has_low_ece():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0
