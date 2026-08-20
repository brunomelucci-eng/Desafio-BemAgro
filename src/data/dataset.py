"""Dataset e DataLoader PyTorch para segmentação de fileiras.

Gera pares (image_tensor, mask_tensor) em tempo real a partir do
gerador sintético. Suporta splits train/val/test com seeds disjuntos
para garantir que os mesmos cenários não apareçam em dois splits.

Uso rápido
----------
    from src.data.dataset import make_dataloaders
    train_dl, val_dl = make_dataloaders(
        train_samples=10_000,
        val_samples=1_000,
        image_size=256,
        batch_size=8,
        seed=42,
    )
    for images, masks in train_dl:
        # images: (B, 3, 256, 256) float32
        # masks:  (B, 1, 256, 256) float32
        ...
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.rasterize import rasterize_lines_as_mask, rasterize_points
from src.data.synthetic import SyntheticScene, SyntheticSceneGenerator

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RowSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset PyTorch de segmentação de fileiras geoespaciais.

    Cada item é um par (image, mask):
    - image: tensor float32 (3, H, W) — canais occupancy, density, distance
    - mask:  tensor float32 (1, H, W) — máscara binária das fileiras GT

    O dataset é gerado lazily: cada cena é criada no __getitem__ com seed
    determinístico baseado no índice. Isso significa que o mesmo índice
    sempre retorna a mesma cena, garantindo reprodutibilidade.

    Parameters
    ----------
    n_samples:
        Número de amostras neste split.
    image_size:
        H = W do tensor de saída em pixels.
    seed_offset:
        Deslocamento de seed para separar os splits (train=0, val=10^6, test=2*10^6).
    line_thickness_px:
        Espessura da máscara GT em pixels (3–5 recomendado).
    augment:
        Se True, aplica augmentations (flip, rotação). Usar apenas no treino.
    """

    def __init__(
        self,
        n_samples: int,
        image_size: int = 256,
        seed_offset: int = 0,
        line_thickness_px: int = 3,
        augment: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed_offset = seed_offset
        self.line_thickness_px = line_thickness_px
        self.augment = augment

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Seed completamente isolado por índice + offset do split.
        # Cada chamada cria um gerador independente: o mesmo idx sempre
        # produz exatamente a mesma cena, independente da ordem de acesso.
        scene_seed = self.seed_offset + idx
        gen = SyntheticSceneGenerator(seed=scene_seed)
        # generate() avança o rng interno; como criamos um novo gerador
        # com seed fixo, a primeira chamada sempre dará o mesmo resultado.
        scene: SyntheticScene = gen.generate()

        # Rasterizar pontos (3 canais).
        # Passamos point_spacing_hint do config da cena para que o pixel_size
        # seja calculado deterministicamente sem depender de estimativa NN.
        result = rasterize_points(
            scene.points_xy,
            image_size=self.image_size,
            point_spacing_hint=scene.config.point_spacing,
        )

        # Rasterizar máscara de ground truth (1 canal)
        mask = rasterize_lines_as_mask(
            scene.lines_xy,
            transform=result.transform,
            image_size=self.image_size,
            line_thickness_px=self.line_thickness_px,
        )

        image_t = torch.from_numpy(result.tensor)  # (3, H, W)
        mask_t = torch.from_numpy(mask)             # (1, H, W)

        # Augmentations coerentes entre image e mask.
        # Generator local com seed derivado do índice garante que o mesmo
        # idx sempre produza o mesmo aug, independente da ordem de acesso.
        if self.augment:
            aug_seed = scene_seed ^ 0xDEADBEEF  # XOR para separar do scene_seed
            aug_gen = torch.Generator().manual_seed(aug_seed & 0xFFFFFFFF)
            image_t, mask_t = _augment(image_t, mask_t, aug_gen)

        return image_t, mask_t


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _augment(
    image: torch.Tensor,
    mask: torch.Tensor,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica transformações aleatórias coerentes entre image e mask.

    As augmentations são:
    - Flip horizontal (p=0.5)
    - Flip vertical (p=0.5)
    - Rotação de 90° (p=0.5 — múltiplo de 90° para preservar estrutura)

    Não usamos valores fora de {0°, 90°, 180°, 270°} para evitar
    introduzir artefatos de interpolação na máscara binária.

    Parameters
    ----------
    rng:
        Generator local com seed fixo por índice. Garante que o mesmo item
        do dataset sempre receba exatamente o mesmo aug.
        Se None, usa o estado global do PyTorch (não determinístico).
    """
    # Flip horizontal
    if torch.rand(1, generator=rng).item() > 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # Flip vertical
    if torch.rand(1, generator=rng).item() > 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    # Rotação de 90°
    if torch.rand(1, generator=rng).item() > 0.5:
        k = int(torch.randint(1, 4, (1,), generator=rng).item())
        image = torch.rot90(image, k=k, dims=[1, 2])
        mask = torch.rot90(mask, k=k, dims=[1, 2])

    return image, mask


# ---------------------------------------------------------------------------
# Factory de DataLoaders
# ---------------------------------------------------------------------------

# Seed offsets para cada split — garantem que os índices nunca se sobreponham
_SEED_OFFSETS: dict[Split, int] = {
    "train": 0,
    "val": 10_000_000,
    "test": 20_000_000,
}


def make_dataset(
    split: Split,
    n_samples: int,
    image_size: int = 256,
    line_thickness_px: int = 3,
    augment: bool | None = None,
) -> RowSegmentationDataset:
    """Cria um RowSegmentationDataset para o split especificado.

    Parameters
    ----------
    split:
        "train", "val" ou "test".
    n_samples:
        Número de amostras no split.
    image_size:
        Resolução H=W do raster em pixels.
    line_thickness_px:
        Espessura da máscara GT em pixels.
    augment:
        Se None, augmentations são ativadas apenas no split "train".
    """
    do_augment = augment if augment is not None else (split == "train")
    return RowSegmentationDataset(
        n_samples=n_samples,
        image_size=image_size,
        seed_offset=_SEED_OFFSETS[split],
        line_thickness_px=line_thickness_px,
        augment=do_augment,
    )


def make_dataloaders(
    train_samples: int = 10_000,
    val_samples: int = 1_000,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    line_thickness_px: int = 3,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Cria DataLoaders de treino e validação prontos para uso.

    Parameters
    ----------
    train_samples, val_samples:
        Número de amostras em cada split.
    image_size:
        Resolução H=W dos tensores.
    batch_size:
        Tamanho do batch.
    num_workers:
        Workers para carregamento paralelo. 0 = thread principal (mais simples
        em Windows).
    line_thickness_px:
        Espessura da máscara GT.
    seed:
        Seed do DataLoader (para shuffle reproduzível).

    Returns
    -------
    (train_dataloader, val_dataloader)
    """
    train_ds = make_dataset("train", train_samples, image_size, line_thickness_px)
    val_ds = make_dataset("val", val_samples, image_size, line_thickness_px, augment=False)

    generator = torch.Generator().manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "DataLoaders prontos: train=%d amostras, val=%d amostras, batch_size=%d",
        train_samples,
        val_samples,
        batch_size,
    )
    return train_dl, val_dl
