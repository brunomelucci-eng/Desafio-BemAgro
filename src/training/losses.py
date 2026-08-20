"""Funções de loss para segmentação de linhas finas.

Problema de desbalanceamento
-----------------------------
Em segmentação de fileiras agrícolas, as linhas ocupam tipicamente
5–15% dos pixels da imagem. Isso gera um forte desbalanceamento
de classes: ~85% de pixels negativos (fundo) vs ~15% positivos (linha).

Usar apenas Binary Cross Entropy padrão resultaria num modelo que
aprende a prever sempre "fundo" e obtém ~85% de accuracy sem aprender
nada. Por isso combinamos:

1. BCEWithLogitsLoss:
   Garante gradiente denso e numericamente estável (combina sigmoid
   e BCE num único passo para evitar overflow).

2. Dice Loss:
   Otimiza diretamente o coeficiente de similaridade entre predição
   e ground truth. É invariante ao desbalanceamento de classes porque
   calcula a razão de sobreposição, não a contagem absoluta de pixels.

Loss total: total = bce_weight × BCE + dice_weight × Dice

Suporte a deep supervision
---------------------------
Quando o modelo usa deep_supervision=True, a forward retorna uma lista
de tensores. CombinedLoss calcula a loss sobre cada saída e retorna
a média, permitindo que os nós intermediários da U-Net++ também
recebam gradiente direto.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss para segmentação binária.

    Dice = 2 × |P ∩ G| / (|P| + |G|)
    DiceLoss = 1 − Dice

    O smooth evita divisão por zero quando predição e GT são ambos zeros.
    Aplicamos sigmoid internamente para aceitar logits como entrada,
    consistente com BCEWithLogitsLoss.

    Parameters
    ----------
    smooth:
        Constante de suavização para estabilidade numérica (default 1.0).
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula Dice Loss.

        Parameters
        ----------
        logits: Tensor (B, 1, H, W) — saída bruta da rede (antes do sigmoid)
        targets: Tensor (B, 1, H, W) — máscara GT em {0.0, 1.0}

        Returns
        -------
        Tensor escalar — Dice Loss média do batch
        """
        preds = torch.sigmoid(logits)
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Interseção: pontos onde predição e GT são ambos positivos
        intersection = (preds_flat * targets_flat).sum(dim=1)
        # União suavizada
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_sample.mean()


class CombinedLoss(nn.Module):
    """Combinação de BCEWithLogitsLoss + DiceLoss.

    Esta combinação é bem estabelecida em segmentação médica e geoespacial:
    - BCE fornece gradiente pixel-a-pixel com estabilidade numérica
    - Dice penaliza o desequilíbrio entre classes

    Suporta deep_supervision: quando a entrada for uma lista de tensores,
    calcula a loss em cada saída e retorna a média.

    Parameters
    ----------
    bce_weight:
        Peso da componente BCE (default 1.0).
    dice_weight:
        Peso da componente Dice (default 1.0).
    pos_weight:
        Peso para a classe positiva no BCE. Útil para desbalanceamento
        extremo (ex: pos_weight=5 significa que falso-negativos custam
        5× mais que falso-positivos). None = sem peso.
    smooth:
        Smooth da Dice Loss.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth)

        # pos_weight como tensor para BCEWithLogitsLoss
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw)

    def _compute_single(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Loss para um único par (logits, targets)."""
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

    def forward(
        self,
        logits: torch.Tensor | list[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula a loss combinada.

        Parameters
        ----------
        logits:
            Tensor (B, 1, H, W) ou lista de tensores (deep supervision).
        targets:
            Tensor (B, 1, H, W) com ground truth binário.
        """
        if isinstance(logits, list):
            # Deep supervision: média das losses de todas as saídas
            losses = [self._compute_single(l, targets) for l in logits]
            return torch.stack(losses).mean()
        return self._compute_single(logits, targets)
