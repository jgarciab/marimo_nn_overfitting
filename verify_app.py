"""Lightweight checks for the simplified neural-network teaching app."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


APP = Path("app.py")


def test_app_parses() -> None:
    ast.parse(APP.read_text(encoding="utf-8"))


def test_app_keeps_the_requested_scope() -> None:
    source = APP.read_text(encoding="utf-8")
    assert "make_xor_data" in source
    assert "Self-check" not in source
    assert "Two moons" not in source
    assert "App map" not in source
    assert "Section 3" not in source
    assert "Section 4" not in source
    assert "early_stopping" not in source
    assert "Use early stopping" not in source
    assert "label=\"Seed\"" not in source
    assert "label=\"Epochs\"" in source
    assert "capacity_seed" not in source
    assert "capacity_epochs" in source
    assert "mo.ui.slider(40, 200" in source
    assert "value=100, label=\"Sample size\"" in source
    assert "value=0.2, label=\"Noise\"" in source
    assert "_epochs = capacity_epochs.value" in source
    assert "#5e548e" in source
    assert "#c9a227" in source
    assert "l1_strength" not in source
    assert "l2_strength" in source
    assert "plot_boundary_comparison" in source
    assert "Dropout + L2" in source
    assert "training, regularized" not in source
    assert "Correct model error" in source
    assert "true XOR boundary" in source
    assert "ax.axhline(" in source
    assert "ax.axvline(" in source
    assert "fitted parameter count" in source
    assert "parameter_breakdown" in source
    assert "make_metric_rows" in source
    assert "add_gridspec(1, 2" in source


def test_parameter_count_formula() -> None:
    dims = [2] + [128] * 6 + [1]
    param_count = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    assert param_count == 83073


def test_sigmoid_and_loss_are_finite() -> None:
    z = np.linspace(-40, 40, 200)
    p = 1 / (1 + np.exp(-z))
    y = (z > 0).astype(float)
    loss = -np.mean(
        y * np.log(np.clip(p, 1e-7, 1))
        + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))
    )
    assert np.isfinite(p).all()
    assert np.isfinite(loss)


if __name__ == "__main__":
    test_app_parses()
    test_app_keeps_the_requested_scope()
    test_parameter_count_formula()
    test_sigmoid_and_loss_are_finite()
    print("Simplified neural-network app checks passed.")
