"""Blocos construtores da U-Net++.

Todos os módulos são stateless (sem estado global), parametrizados,
e seguem o padrão nn.Module do PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloco de convolução dupla: (Conv → BN → ReLU) × 2.

    É o bloco fundamental da U-Net e da U-Net++. A aplicação dupla
    de convolução permite que cada bloco aprenda transformações mais
    ricas sem aumentar excessivamente a profundidade da rede.

    Parameters
    ----------
    in_channels:
        Número de canais de entrada.
    out_channels:
        Número de canais de saída (= número de filtros).
    dropout_p:
        Probabilidade de dropout aplicada após a segunda ativação.
        0.0 desabilita o dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            # Primeira convolução 3×3 — extrai features locais
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Segunda convolução 3×3 — refina as features extraídas
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            # Dropout aplicado após a segunda ativação para regularização
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Bloco de encoder: MaxPool 2×2 seguido de ConvBlock.

    O MaxPool reduz H e W pela metade (downsampling), aumentando o
    campo receptivo. O ConvBlock extrai features na nova resolução.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bloco de decoder: upsampling bilinear + concatenação + ConvBlock.

    O upsampling bilinear é preferível à TransposeConv para evitar o
    artefato de "checkerboard" (xadrez) comum em decodificadores.
    A concatenação implementa a skip connection da U-Net.

    Parameters
    ----------
    in_channels:
        Canais que chegam pelo caminho de upsampling.
    skip_channels:
        Soma dos canais de TODAS as skip connections concatenadas.
        Na U-Net++ isso pode ser maior que na U-Net simples, pois
        múltiplos nós intermediários são concatenados.
    out_channels:
        Canais de saída do bloco.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Concatenar todas as skip connections ao longo da dimensão de canais
        x = torch.cat([x, *skips], dim=1)
        return self.conv(x)
