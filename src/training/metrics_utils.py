from __future__ import annotations

from typing import Dict, Tuple, List, Literal, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def _safe_roc_auc_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Calcula ROC-AUC multi-classe (OVR) de forma defensiva.
    Retorna NaN se não for possível (ex.: uma única classe presente).
    """
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average=average,
            )
        )
    except Exception:
        return float("nan")


def compute_patch_metrics_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Métricas em nível de PATCH para problema multi-classe (estágios I–IV).
    - y_true: shape [N], ints
    - y_proba: shape [N, C], probs (softmax)
    """
    y_pred = y_proba.argmax(axis=1)

    metrics: Dict[str, float] = {}
    metrics["patch_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["patch_f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))

    # F1 por classe
    f1_per_class = cast(
        np.ndarray,
        f1_score(y_true, y_pred, average=None),
    )
    for index, cname in enumerate(class_names):
        metrics[f"patch_f1_{cname}"] = float(f1_per_class[index])

    # ROC-AUC multi-classe macro
    metrics["patch_auc_roc_macro"] = _safe_roc_auc_multiclass(y_true, y_proba, "macro")

    return metrics


def aggregate_by_slide(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    slide_ids: np.ndarray,
    method: Literal["mean_prob", "majority_vote"] = "mean_prob",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Agrega previsões por slide_id.

    method = "mean_prob":
        - probabilidade média dos patches (soft voting)
        - ground truth do slide = moda de y_true (por segurança)

    method = "majority_vote":
        - prob slide-level = distribuição de frequências das predições (hard voting)
        - ground truth do slide = moda de y_true
    """
    slide_ids = np.asarray(slide_ids)
    unique_slides = np.unique(slide_ids)

    y_true_slide_list: List[int] = []
    y_proba_slide_list: List[np.ndarray] = []

    y_pred_patch = y_proba.argmax(axis=1)
    num_classes = y_proba.shape[1]

    for slide in unique_slides:
        idx = np.where(slide_ids == slide)[0]
        ys = y_true[idx]
        vals, counts = np.unique(ys, return_counts=True)
        slide_label_true = int(vals[counts.argmax()])

        if method == "mean_prob":
            proba_mean = y_proba[idx].mean(axis=0)
        elif method == "majority_vote":
            preds = y_pred_patch[idx]
            counts_pred = np.bincount(preds, minlength=num_classes).astype(float)
            proba_mean = counts_pred / counts_pred.sum()
        else:
            raise ValueError(f"Método de agregação não suportado: {method}")

        y_true_slide_list.append(slide_label_true)
        y_proba_slide_list.append(proba_mean)

    return np.array(y_true_slide_list), np.vstack(y_proba_slide_list)


def compute_slide_metrics_multiclass(
    y_true_slide: np.ndarray,
    y_proba_slide: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Métricas em nível de SLIDE para problema multi-classe.
    """
    y_pred_slide = y_proba_slide.argmax(axis=1)

    metrics: Dict[str, float] = {}
    metrics["slide_accuracy"] = float(accuracy_score(y_true_slide, y_pred_slide))
    metrics["slide_f1_macro"] = float(
        f1_score(y_true_slide, y_pred_slide, average="macro")
    )

    f1_per_class = cast(
        np.ndarray,
        f1_score(y_true_slide, y_pred_slide, average=None),
    )
    for index, cname in enumerate(class_names):
        metrics[f"slide_f1_{cname}"] = float(f1_per_class[index])

    metrics["slide_auc_roc_macro"] = _safe_roc_auc_multiclass(
        y_true_slide, y_proba_slide, "macro"
    )

    return metrics


def dump_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Wrapper para sklearn.classification_report, útil para logging/debug.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=False,
    )
    return cast(str, report)
