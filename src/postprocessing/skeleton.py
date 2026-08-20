"""Skeletonização e pós-processamento morfológico da máscara de segmentação.

Pipeline de pós-processamento
-------------------------------
probability_map (float)
    → threshold → máscara binária
    → morfologia (remove ruído)
    → skeletonização (linhas de 1 pixel)
    → remoção de componentes pequenos
    → máscara esqueleto limpa

Por que skeletonização é necessária
-------------------------------------
A U-Net++ produz uma máscara de N pixels de espessura (ex: 3–5 px).
Para vetorizar como LineString, precisamos de uma representação de
1 pixel de espessura, onde cada pixel representa exatamente um ponto
da linha centerline. O algoritmo de Zhang-Suen (scikit-image) faz isso
preservando a conectividade topológica.

Dependência
-----------
scikit-image é necessário. Ele não está no requirements.txt original,
mas será adicionado junto com as demais dependências do pipeline ML.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# Área mínima de um componente conectado para não ser removido (pixels)
MIN_COMPONENT_AREA_PX: int = 10


def probability_to_skeleton(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_area_px: int = MIN_COMPONENT_AREA_PX,
) -> np.ndarray:
    """Converte mapa de probabilidades em esqueleto binário de 1 pixel.

    Parameters
    ----------
    prob_map: ndarray (H, W) float — saída sigmoid da U-Net++ em [0, 1]
    threshold: limiar de binarização (default 0.5)
    min_area_px: componentes com área menor são removidos como ruído

    Returns
    -------
    ndarray bool (H, W) — esqueleto binário (True = pixel de linha)
    """
    # 1. Binarizar
    binary = prob_map >= threshold

    # 2. Limpeza morfológica — remove pequenos objetos e preenche buracos
    binary = _morphological_cleanup(binary, min_area_px)

    # 3. Skeletonização — reduz a linhas de 1 pixel preservando topologia
    skeleton = _skeletonize(binary)

    # 4. Remover componentes extremamente curtos no esqueleto
    skeleton = _remove_short_components(skeleton, min_area_px=3)

    return skeleton


def _morphological_cleanup(binary: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove pequenos objetos e preenche buracos na máscara binária."""
    # Fechamento morfológico: preenche pequenas lacunas dentro das linhas
    struct = ndi.generate_binary_structure(2, 2)  # conectividade 8
    closed = ndi.binary_closing(binary, structure=struct, iterations=1)

    # Rotular componentes conectados
    labeled, n_components = ndi.label(closed)
    if n_components == 0:
        return np.zeros_like(binary)

    # Calcular área de cada componente
    component_sizes = np.bincount(labeled.ravel())
    # Manter apenas componentes com área >= min_area_px
    mask_sizes = component_sizes >= min_area_px
    mask_sizes[0] = False  # background

    # Reconstruir máscara com apenas os componentes grandes
    return mask_sizes[labeled]


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Aplica skeletonização (Zhang-Suen / Lee) para obter linhas de 1 pixel."""
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        return sk_skeletonize(binary)
    except ImportError:
        # Fallback simples se scikit-image não estiver disponível:
        # erosão iterativa até convergir (mais lento, menos preciso)
        logger.warning("scikit-image não encontrado. Usando fallback de erosão.")
        return _erosion_skeleton_fallback(binary)


def _erosion_skeleton_fallback(binary: np.ndarray) -> np.ndarray:
    """Esqueleto por erosão iterativa — fallback sem scikit-image."""
    skeleton = binary.copy()
    struct = ndi.generate_binary_structure(2, 2)
    while True:
        eroded = ndi.binary_erosion(skeleton, structure=struct)
        # Parar quando a erosão adicional destruiria a conectividade
        opened = ndi.binary_dilation(eroded, structure=struct)
        subtracted = skeleton & ~opened
        skeleton_new = skeleton ^ subtracted
        if np.array_equal(skeleton_new, skeleton):
            break
        skeleton = skeleton_new
    return skeleton


def _remove_short_components(skeleton: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove componentes do esqueleto com menos de min_area_px pixels."""
    labeled, _ = ndi.label(skeleton)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    return keep[labeled]
