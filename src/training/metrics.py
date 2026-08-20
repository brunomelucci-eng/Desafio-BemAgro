"""Métricas de avaliação para segmentação binária.

Todas as métricas operam em logits (antes do sigmoid) ou probabilidades,
e aceitam tanto tensores PyTorch quanto arrays NumPy.

Métricas implementadas
----------------------
- Dice Score (F1)
- IoU (Intersection over Union / Jaccard Index)
- Precision
- Recall
- Todas agregadas num dicionário de métricas por conveniência
"""

from __future__ import annotations

import numpy as np
import torch


def _to_binary(
    logits_or_probs: torch.Tensor, threshold: float = 0.5, is_logit: bool = True
) -> torch.Tensor:
    """Converte logits ou probabilidades para máscara binária."""
    if is_logit:
        probs = torch.sigmoid(logits_or_probs)
    else:
        probs = logits_or_probs
    return (probs >= threshold).float()


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Dice Score (F1) entre predição e ground truth.

    Dice = 2 × TP / (2 × TP + FP + FN)

    Parameters
    ----------
    logits: Tensor (B, 1, H, W) — saída bruta da rede
    targets: Tensor (B, 1, H, W) — máscara GT binária
    threshold: limiar para binarização das probabilidades
    smooth: constante para estabilidade numérica

    Returns
    -------
    Dice Score escalar em [0, 1] (maior = melhor)
    """
    preds = _to_binary(logits, threshold)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    return float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Intersection over Union (IoU / Jaccard Index).

    IoU = TP / (TP + FP + FN)

    Sempre menor ou igual ao Dice para o mesmo par (predição, GT).
    """
    preds = _to_binary(logits, threshold)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    return float((tp + smooth) / (tp + fp + fn + smooth))


def precision_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Precision = TP / (TP + FP).

    Quanto dos pixels preditos como linha são realmente linha.
    """
    preds = _to_binary(logits, threshold)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def recall_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    """Recall = TP / (TP + FN).

    Quanto das linhas reais foram detectadas pela rede.
    """
    preds = _to_binary(logits, threshold)
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula todas as métricas de segmentação de uma vez.

    Parameters
    ----------
    logits: Tensor (B, 1, H, W)
    targets: Tensor (B, 1, H, W)
    threshold: limiar de binarização

    Returns
    -------
    dict com: dice, iou, precision, recall
    """
    # Quando deep supervision, usar apenas a última saída
    if isinstance(logits, list):
        logits = logits[-1]

    return {
        "dice": dice_score(logits, targets, threshold),
        "iou": iou_score(logits, targets, threshold),
        "precision": precision_score(logits, targets, threshold),
        "recall": recall_score(logits, targets, threshold),
    }


class MetricTracker:
    """Acumulador de métricas por época.

    Uso:
        tracker = MetricTracker()
        for batch in loader:
            tracker.update(loss=0.3, dice=0.85, iou=0.74)
        epoch_metrics = tracker.compute()
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, **kwargs: float) -> None:
        """Adiciona valores de uma iteração."""
        for key, value in kwargs.items():
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._counts[key] = self._counts.get(key, 0) + 1

    def compute(self) -> dict[str, float]:
        """Retorna a média de cada métrica acumulada."""
        return {
            key: self._sums[key] / max(self._counts[key], 1)
            for key in self._sums
        }

    def reset(self) -> None:
        """Limpa o acumulador para a próxima época."""
        self._sums.clear()
        self._counts.clear()
