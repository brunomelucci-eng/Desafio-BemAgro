"""Vetorização: converte esqueleto binário em LineStrings georreferenciadas.

Pipeline
--------
skeleton (bool array H×W)
    → detecção de componentes conectados
    → construção de grafo de pixels (8-vizinhança)
    → poda de ramificações espúrias
    → ordenação de pixels ao longo da linha (DFS greedy)
    → criação de LineString
    → conversão pixel → coordenadas métricas
    → suavização e simplificação
    → validação topológica

Conversão de coordenadas
------------------------
Usa o AffineTransform do módulo rasterize para converter coordenadas
de pixel (col, row) de volta para (x, y) métricas (UTM), e depois
o CRS do GeoDataFrame original para reprojetar para EPSG:4326.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import geopandas as gpd
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString

from src.data.rasterize import AffineTransform, pixel_to_world

logger = logging.getLogger(__name__)

# Comprimento mínimo de uma linha vetorizada (em pixels de esqueleto)
MIN_LINE_LENGTH_PX: int = 5


def skeleton_to_linestrings(
    skeleton: np.ndarray,
    transform: AffineTransform,
    metric_crs: str,
    min_length_px: int = MIN_LINE_LENGTH_PX,
    simplify_tolerance: float = 0.5,
) -> gpd.GeoDataFrame:
    """Converte esqueleto binário em GeoDataFrame de LineStrings métricas.

    Parameters
    ----------
    skeleton: ndarray bool (H, W) — saída do módulo skeleton
    transform: AffineTransform para conversão pixel → metros
    metric_crs: string EPSG do CRS métrico (ex: "EPSG:32723")
    min_length_px: linhas com menos pixels que isso são descartadas
    simplify_tolerance: tolerância de Douglas-Peucker em metros

    Returns
    -------
    GeoDataFrame em ``metric_crs`` com coluna ``geometry`` (LineString)
    e colunas auxiliares ``n_pixels``, ``length_m``.
    """
    # 0. Quebrar o esqueleto nas bifurcações (junctions)
    # Isso evita que linhas se cruzem formando um único componente em zigue-zague
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    neighbors = ndi.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    junctions = (neighbors >= 3) & skeleton
    broken_skeleton = skeleton & ~junctions

    # 1. Identificar componentes conectados no esqueleto quebrado
    labeled, n_components = ndi.label(broken_skeleton, structure=np.ones((3, 3)))
    if n_components == 0:
        logger.warning("Esqueleto vazio — nenhuma linha encontrada.")
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)

    lines: list[LineString] = []
    n_pixels_list: list[int] = []

    for comp_id in range(1, n_components + 1):
        component_mask = labeled == comp_id
        pixel_coords = np.column_stack(np.where(component_mask))  # (row, col)

        if len(pixel_coords) < min_length_px:
            continue

        # Converter (row, col) → (col, row) para world_to_pixel convention
        px_col_row = pixel_coords[:, [1, 0]].astype(float)  # (N, 2): col, row

        # 2. Ordenar pixels ao longo da linha (caminhada greedy)
        ordered_px = _order_pixels(pixel_coords, component_mask)
        if len(ordered_px) < 2:
            continue

        # Converter para (col, row) antes de chamar pixel_to_world
        ordered_col_row = ordered_px[:, [1, 0]].astype(float)

        # 3. Converter pixel → coordenadas métricas
        world_coords = pixel_to_world(ordered_col_row, transform)

        # 4. Criar LineString e simplificar
        line = LineString(world_coords)
        if not line.is_valid or line.length < simplify_tolerance:
            continue

        if simplify_tolerance > 0:
            line = line.simplify(simplify_tolerance, preserve_topology=True)

        lines.append(line)
        n_pixels_list.append(len(ordered_px))

    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)

    gdf = gpd.GeoDataFrame(
        {
            "n_pixels": n_pixels_list,
            "length_m": [g.length for g in lines],
            "geometry": lines,
        },
        crs=metric_crs,
    )
    return gdf


def _order_pixels(
    pixel_coords: np.ndarray, component_mask: np.ndarray
) -> np.ndarray:
    """Ordena pixels de um componente ao longo da linha (DFS greedy).

    Começa pela extremidade com menos vizinhos (endpoint) e avança
    sempre para o vizinho não-visitado mais próximo.

    Parameters
    ----------
    pixel_coords: ndarray (N, 2) — coordenadas (row, col) do componente
    component_mask: ndarray bool (H, W) — máscara do componente

    Returns
    -------
    ndarray (M, 2) — pixels ordenados ao longo da linha (row, col)
    """
    if len(pixel_coords) <= 2:
        return pixel_coords

    H, W = component_mask.shape

    # Construir grafo de vizinhança 8-conectado
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    pixel_set = set(map(tuple, pixel_coords))

    for r, c in pixel_coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in pixel_set:
                    adjacency[(r, c)].append((nr, nc))

    # Encontrar endpoints (nós com grau 1) para começar a ordenação
    endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]

    # Se não há endpoints, escolher qualquer ponto (componente circular)
    start = endpoints[0] if endpoints else tuple(pixel_coords[0])

    # DFS greedy: avançar sempre para o vizinho não-visitado
    visited: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    stack: list[tuple[int, int]] = [start]  # type: ignore[list-item]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        ordered.append(current)
        # Adicionar vizinhos não-visitados
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                stack.append(neighbor)

    return np.array(ordered)
