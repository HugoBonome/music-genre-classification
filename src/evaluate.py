"""Evaluation metrics and plots shared across the modeling notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config as C


def evaluate_model(model, X, y_true) -> dict:
    """Predict and return accuracy, macro F1 and predicted labels.

    ``X`` is cast to float32 (cached spectrograms are float16).
    """
    probs = model.predict(X.astype(np.float32), verbose=0)
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "y_pred": y_pred,
        "y_prob": probs,
    }


def print_report(y_true, y_pred, genres=C.GENRES) -> None:
    print(classification_report(y_true, y_pred, target_names=genres, digits=3))


def plot_confusion_matrix(y_true, y_pred, genres=C.GENRES, normalize=True, title="Matriz de confusão"):
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=genres,
                yticklabels=genres, cbar=True, ax=ax)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_history(history, title="Curvas de treino"):
    h = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(h["loss"], label="treino")
    axes[0].plot(h["val_loss"], label="validação")
    axes[0].set_title("Loss"); axes[0].set_xlabel("época"); axes[0].legend()
    axes[1].plot(h["accuracy"], label="treino")
    axes[1].plot(h["val_accuracy"], label="validação")
    axes[1].set_title("Acurácia"); axes[1].set_xlabel("época"); axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    return axes


def save_history(history, path: Path) -> None:
    h = history.history if hasattr(history, "history") else history
    Path(path).write_text(json.dumps({k: [float(v) for v in vs] for k, vs in h.items()}))


def append_result(name: str, accuracy: float, f1_macro: float,
                  path: Path = C.MODELS_DIR / "results.json") -> dict:
    """Append/update one experiment's metrics in a shared results.json table."""
    path = Path(path)
    results = json.loads(path.read_text()) if path.exists() else {}
    results[name] = {"accuracy": float(accuracy), "f1_macro": float(f1_macro)}
    path.write_text(json.dumps(results, indent=2))
    return results
