"""Rasterização de pontos geoespaciais para tensores PyTorch.

Converte um conjunto de pontos métricos (UTM) em um tensor multi-canal
adequado para segmentação semântica com U-Net++.

Canais produzidos
-----------------
ch0 — Occupancy map:
    Presença binária de pontos. Cada ponto projeta num pixel com
    antialiasing gaussiano leve (sigma = 0.5 px) para evitar
    sinais de largura zero.

ch1 — Density map (KDE aproximado):
    Convolução gaussiana de ch0 com sigma maior (sigma = 1.5 px).
    Captura a vizinhança e densidade local dos pontos.

ch2 — Distance map (invertida e normalizada):
    Distância euclidiana ao ponto mais próximo, normalizada em
    [0, 1] e invertida (1 = muito perto, 0 = muito longe).
    Fornece gradiente contínuo útil para rastrear fileiras.

Transformação espacial
----------------------
A ``AffineTransform`` (origin_x, origin_y, pixel_size) permite
converter coordenadas de pixel de volta para metros e depois para
EPSG:4326 na etapa de vetorização.

Resolução adaptativa
--------------------
``pixel_size = point_spacing / resolution_factor``
Por padrão ``resolution_factor = 4``, o que coloca ~4 pixels entre
pontos consecutivos. Isso é calculado automaticamente a partir dos
dados ou passado explicitamente.

Uso
---
    from src.data.rasterize import rasterize_points, world_to_pixel, pixel_to_world
    tensor, transform = rasterize_points(points_xy, pixel_size=0.5, image_size=256)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

logger = logging.getLogger(__name__)

# Fator padrão: 4 pixels por espaçamento entre pontos
DEFAULT_RESOLUTION_FACTOR: int = 4
# Padding ao redor do bounding box dos pontos (em pixels)
DEFAULT_PADDING_PX: int = 8


# ---------------------------------------------------------------------------
# Transformação espacial
# ---------------------------------------------------------------------------

class AffineTransform(NamedTuple):
    """Transformação afim pixel ↔ coordenadas métricas (sem rotação).

    Attributes
    ----------
    origin_x, origin_y:
        Coordenadas métricas do canto superior-esquerdo do raster (pixel 0,0).
    pixel_size:
        Tamanho do pixel em metros. Pixels quadrados, sem rotação.
    """

    origin_x: float
    origin_y: float
    pixel_size: float


def world_to_pixel(xy: np.ndarray, transform: AffineTransform) -> np.ndarray:
    """Converte coordenadas métricas (x, y) para (col, row) em pixels.

    Note que ``row`` cresce para baixo (convenção de imagem): quanto maior
    o y no mundo, menor o row.

    Parameters
    ----------
    xy: ndarray (N, 2) — coordenadas métricas (x=leste, y=norte)
    transform: AffineTransform com origin e pixel_size

    Returns
    -------
    ndarray (N, 2) — (col, row) em pixels (float, sem arredondar)
    """
    col = (xy[:, 0] - transform.origin_x) / transform.pixel_size
    row = (transform.origin_y - xy[:, 1]) / transform.pixel_size  # y invertido
    return np.column_stack([col, row])


def pixel_to_world(px: np.ndarray, transform: AffineTransform) -> np.ndarray:
    """Converte (col, row) em pixels para coordenadas métricas (x, y).

    Inversa de ``world_to_pixel``.
    """
    x = transform.origin_x + px[:, 0] * transform.pixel_size
    y = transform.origin_y - px[:, 1] * transform.pixel_size
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Rasterização principal
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RasterizationResult:
    """Resultado de uma rasterização."""

    tensor: np.ndarray          # float32 (3, H, W) — os 3 canais
    transform: AffineTransform  # para conversão pixel ↔ mundo
    image_size: int             # H = W (imagem quadrada)
    pixel_size: float           # metros por pixel


def rasterize_points(
    points_xy: np.ndarray,
    pixel_size: float | None = None,
    image_size: int = 256,
    point_spacing_hint: float | None = None,
    padding_px: int = DEFAULT_PADDING_PX,
) -> RasterizationResult:
    """Rasteriza pontos métricos (N, 2) em um tensor float32 (3, H, W).

    Parameters
    ----------
    points_xy:
        Array (N, 2) com coordenadas métricas (x, y).
    pixel_size:
        Tamanho do pixel em metros. Se None, é estimado automaticamente a partir
        do ``point_spacing_hint`` ou da mediana das distâncias ao vizinho mais
        próximo.
    image_size:
        Dimensão H = W do tensor de saída (pixels). O campo é cropado/escalado
        para caber nesta resolução.
    point_spacing_hint:
        Espaçamento estimado entre pontos (metros). Usado apenas para calcular
        ``pixel_size`` quando não especificado.
    padding_px:
        Margem em pixels ao redor do bounding box dos pontos.

    Returns
    -------
    RasterizationResult com tensor (3, H, W) e AffineTransform.
    """
    if len(points_xy) == 0:
        # Retorna tensor vazio com transformação unitária
        tensor = np.zeros((3, image_size, image_size), dtype=np.float32)
        transform = AffineTransform(origin_x=0.0, origin_y=float(image_size), pixel_size=1.0)
        return RasterizationResult(tensor, transform, image_size, 1.0)

    # Estimar pixel_size automaticamente
    if pixel_size is None:
        pixel_size = _estimate_pixel_size(points_xy, point_spacing_hint, image_size, padding_px)

    # Calcular origem do raster (canto superior-esquerdo)
    min_x = float(points_xy[:, 0].min())
    max_y = float(points_xy[:, 1].max())
    origin_x = min_x - padding_px * pixel_size
    origin_y = max_y + padding_px * pixel_size  # origem no topo (y positivo = norte)

    transform = AffineTransform(origin_x=origin_x, origin_y=origin_y, pixel_size=pixel_size)

    # Calcular tamanho necessário
    px_coords = world_to_pixel(points_xy, transform)
    needed_cols = int(np.ceil(px_coords[:, 0].max())) + padding_px + 1
    needed_rows = int(np.ceil(px_coords[:, 1].max())) + padding_px + 1

    # Usar image_size fixo: escalar se necessário
    if needed_cols > image_size or needed_rows > image_size:
        # Reescalar pixel_size para que tudo caiba em image_size
        scale = max(needed_cols, needed_rows) / (image_size - 2 * padding_px)
        pixel_size = pixel_size * scale
        origin_x = min_x - padding_px * pixel_size
        origin_y = max_y + padding_px * pixel_size
        transform = AffineTransform(origin_x=origin_x, origin_y=origin_y, pixel_size=pixel_size)
        px_coords = world_to_pixel(points_xy, transform)

    # Construir canvas com tamanho fixo
    H = W = image_size
    ch0 = _build_occupancy_channel(px_coords, H, W)
    ch1 = _build_density_channel(ch0)
    ch2 = _build_distance_channel(ch0)

    tensor = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

    return RasterizationResult(tensor=tensor, transform=transform, image_size=image_size, pixel_size=pixel_size)


def rasterize_lines_as_mask(
    lines_xy: list[np.ndarray],
    transform: AffineTransform,
    image_size: int,
    line_thickness_px: int = 3,
) -> np.ndarray:
    """Rasteriza linhas de ground truth como máscara binária float32 (1, H, W).

    Parameters
    ----------
    lines_xy:
        Lista de arrays (M_i, 2) com coordenadas métricas das linhas GT.
    transform:
        Mesma transformação usada no rasterize_points correspondente.
    image_size:
        H = W do canvas de saída.
    line_thickness_px:
        Espessura da linha em pixels. Valores entre 3 e 5 melhoram
        a estabilidade do treinamento versus linhas de 1 pixel.

    Returns
    -------
    ndarray float32 (1, H, W) com valores em {0.0, 1.0}.
    """
    H = W = image_size
    mask = np.zeros((H, W), dtype=np.float32)

    for line_pts in lines_xy:
        if len(line_pts) < 2:
            continue
        px = world_to_pixel(line_pts, transform)
        # Rasterizar cada segmento da polilinha
        for i in range(len(px) - 1):
            _draw_line_segment(mask, px[i], px[i + 1], H, W)

    # Dilatar para atingir a espessura desejada
    if line_thickness_px > 1:
        from scipy.ndimage import binary_dilation
        struct_size = line_thickness_px
        struct = np.ones((struct_size, struct_size), dtype=bool)
        mask = binary_dilation(mask > 0, structure=struct).astype(np.float32)

    return mask[np.newaxis, :, :]  # (1, H, W)


# ---------------------------------------------------------------------------
# Funções auxiliares de canal
# ---------------------------------------------------------------------------

def _build_occupancy_channel(px_coords: np.ndarray, H: int, W: int) -> np.ndarray:
    """Canal 0: presão de pontos (binária sem blur para maior velocidade)."""
    canvas = np.zeros((H, W), dtype=np.float32)
    cols = np.clip(np.round(px_coords[:, 0]).astype(int), 0, W - 1)
    rows = np.clip(np.round(px_coords[:, 1]).astype(int), 0, H - 1)
    canvas[rows, cols] = 1.0
    return canvas


def _build_density_channel(occupancy: np.ndarray) -> np.ndarray:
    """Canal 1: KDE aproximado com gaussiana moderada."""
    density = gaussian_filter(occupancy, sigma=2.0)
    max_val = density.max()
    if max_val > 0:
        density /= max_val
    return density.astype(np.float32)


def _build_distance_channel(occupancy: np.ndarray) -> np.ndarray:
    """Canal 2: distância invertida e normalizada ao ponto mais próximo.

    Pixels com ponto têm valor 1.0 (distância 0). Quanto mais longe
    de qualquer ponto, mais próximo de 0.0.
    """
    # distance_transform_edt mede distância em pixels para o zero mais próximo
    # Aqui "foreground" = sem ponto, então invertemos: background = pontos
    binary = occupancy < 0.5  # True = sem ponto
    dist = distance_transform_edt(binary).astype(np.float32)
    max_dist = dist.max()
    if max_dist > 0:
        dist /= max_dist
    return 1.0 - dist  # inverter: perto = 1, longe = 0


def _draw_line_segment(
    canvas: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    H: int,
    W: int,
) -> None:
    """Rasteriza um segmento de linha usando algoritmo de Bresenham."""
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < H and 0 <= x0 < W:
            canvas[y0, x0] = 1.0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ---------------------------------------------------------------------------
# Estimativa automática de pixel_size
# ---------------------------------------------------------------------------

def _estimate_pixel_size(
    points_xy: np.ndarray,
    point_spacing_hint: float | None,
    image_size: int,
    padding_px: int,
) -> float:
    """Estima pixel_size de forma que o campo caiba em image_size pixels.

    Usa point_spacing_hint se disponível; caso contrário, estima pela mediana
    das distâncias ao vizinho mais próximo (amostrando até 200 pontos).
    """
    if point_spacing_hint is not None:
        spacing = point_spacing_hint
    else:
        spacing = _median_nn_distance(points_xy)

    # pixel_size tal que point_spacing = DEFAULT_RESOLUTION_FACTOR pixels
    pixel_size_from_spacing = spacing / DEFAULT_RESOLUTION_FACTOR

    # Verificar se o campo cabe em image_size com esse pixel_size
    min_x, max_x = float(points_xy[:, 0].min()), float(points_xy[:, 0].max())
    min_y, max_y = float(points_xy[:, 1].min()), float(points_xy[:, 1].max())
    field_w = max_x - min_x
    field_h = max_y - min_y
    usable_px = image_size - 2 * padding_px
    pixel_size_from_field = max(field_w, field_h) / max(usable_px, 1)

    return max(pixel_size_from_spacing, pixel_size_from_field)


def _median_nn_distance(points_xy: np.ndarray) -> float:
    """Mediana das distâncias ao vizinho mais próximo (amostra até 200 pts).

    Usa seed fixo (42) para garantir que a amostragem seja determinística
    e o resultado final de rasterize_points seja reproduzível dado o mesmo input.
    """
    from scipy.spatial import cKDTree

    sample_size = min(200, len(points_xy))
    # seed fixo: garantia de determinismo — mesmos pontos → mesmo pixel_size
    idx = np.random.default_rng(42).choice(len(points_xy), size=sample_size, replace=False)
    sample = points_xy[idx]
    tree = cKDTree(points_xy)
    dists, _ = tree.query(sample, k=2)
    nn_dists = dists[:, 1]  # distância ao 2º vizinho (1º é o próprio ponto)
    return float(np.median(nn_dists[nn_dists > 0]))
