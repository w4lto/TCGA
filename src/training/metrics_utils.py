from __future__ import annotations

from typing import Dict, Tuple, List, Literal, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score
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


def compute_slide_metrics_multiclass(y_true, y_proba, class_names):
    """
    Calcula métricas agregadas por slide (Multiclasse), forçando a presença de todas as classes.
    """
    # Converte probabilidades para predição hard (0, 1, 2, 3)
    y_pred = np.argmax(y_proba, axis=1)
    
    # Índices esperados (ex: [0, 1, 2, 3] para 4 classes)
    labels_indices = list(range(len(class_names)))

    metrics = {}
    
    # 1. Acurácia Global
    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    
    # 2. F1 Macro e Weighted (Globais)
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", labels=labels_indices, zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", labels=labels_indices, zero_division=0))
    
    # 3. Métricas por Classe (Onde o erro ocorria)
    # Ao passar 'labels=labels_indices', o sklearn garante que o array retornado 
    # tenha o tamanho correto, preenchendo com 0 onde não houver predição/ground-truth.
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels_indices, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, labels=labels_indices, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=labels_indices, zero_division=0)
    
    for i, cname in enumerate(class_names):
        metrics[f"slide_f1_{cname}"] = float(f1_per_class[i])
        metrics[f"slide_prec_{cname}"] = float(prec_per_class[i])
        metrics[f"slide_rec_{cname}"] = float(rec_per_class[i])
        
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
