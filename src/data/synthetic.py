"""Gerador sintético supervisionado de cenários geoespaciais.

Filosofia de design
-------------------
O ground truth é criado ANTES dos pontos.

1. Geramos linhas verdadeiras (retas paralelas, curvas polinomiais, arcos).
2. Amostramos pontos ao longo dessas linhas.
3. Adicionamos perturbações, gaps, outliers e ruído.

Dessa forma a supervisão é exata: sabemos exatamente quais pixels devem
ser "linha" no raster target sem depender do algoritmo geométrico como
pseudo-label.

Uso rápido
----------
    from src.data.synthetic import SyntheticSceneGenerator
    gen = SyntheticSceneGenerator(seed=42)
    scene = gen.generate()
    # scene.points_xy  → ndarray (N, 2) em metros
    # scene.lines_xy   → list[ndarray]  ground truth
    # scene.config     → SceneConfig com todos os parâmetros
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos de cenário
# ---------------------------------------------------------------------------
SceneType = Literal["parallel_straight", "parallel_curved", "concentric_arcs"]


@dataclasses.dataclass
class SceneConfig:
    """Todos os parâmetros aleatórios de um cenário sintético."""

    scene_type: SceneType
    n_rows: int
    row_spacing: float        # metros entre fileiras
    point_spacing: float      # metros entre pontos dentro da fileira
    orientation_deg: float    # ângulo das fileiras em graus
    field_length: float       # comprimento total do campo em metros
    noise_sigma: float        # desvio padrão do ruído posicional (metros)
    gap_fraction: float       # fração de pontos removidos (simulando gap)
    gap_max_length: float     # comprimento máximo de um gap em metros
    outlier_fraction: float   # fração de pontos aleatórios adicionados
    curvature: float          # coeficiente quadrático da curva (retas = 0)
    seed: int


@dataclasses.dataclass
class SyntheticScene:
    """Um cenário geoespacial sintético completo."""

    points_xy: np.ndarray        # (N, 2) coordenadas métricas
    lines_xy: list[np.ndarray]   # ground truth: uma lista de arrays (M_i, 2)
    config: SceneConfig
    rng: np.random.Generator     # rng usado — para reprodutibilidade


# ---------------------------------------------------------------------------
# Gerador principal
# ---------------------------------------------------------------------------

class SyntheticSceneGenerator:
    """Gera cenários sintéticos com ground truth explícito.

    Parameters
    ----------
    seed:
        Semente global. Cada chamada a ``generate()`` avança o estado interno,
        garantindo que 10.000 chamadas produzam 10.000 cenas distintas e
        reproduzíveis a partir do mesmo seed raiz.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(self) -> SyntheticScene:
        """Gera um único cenário com parâmetros aleatorizados."""
        config = self._sample_config()
        return self._build_scene(config)

    def iter_scenes(self, n: int) -> Iterator[SyntheticScene]:
        """Gera ``n`` cenas sequencialmente (lazy)."""
        for _ in range(n):
            yield self.generate()

    # ------------------------------------------------------------------
    # Amostragem de configuração (domain randomization)
    # ------------------------------------------------------------------

    def _sample_config(self) -> SceneConfig:
        rng = self._rng

        # Tipo de cena
        scene_type: SceneType = rng.choice(  # type: ignore[assignment]
            ["parallel_straight", "parallel_curved", "concentric_arcs"],
            p=[0.45, 0.35, 0.20],
        )

        # Parâmetros do campo
        n_rows = int(rng.integers(3, 30))
        row_spacing = float(rng.uniform(1.5, 12.0))
        point_spacing = float(rng.uniform(0.5, 4.0))
        orientation_deg = float(rng.uniform(0.0, 180.0))
        field_length = float(rng.uniform(20.0, 300.0))

        # Ruído e gaps
        noise_sigma = float(rng.uniform(0.0, point_spacing * 0.25))
        gap_fraction = float(rng.uniform(0.0, 0.30))
        gap_max_length = float(rng.uniform(point_spacing * 2, point_spacing * 10))

        # Outliers
        outlier_fraction = float(rng.uniform(0.0, 0.15))

        # Curvatura (usada apenas para parallel_curved)
        curvature = float(rng.uniform(0.0005, 0.006)) if scene_type == "parallel_curved" else 0.0

        # Seed filho para esta cena específica
        child_seed = int(rng.integers(0, 2**31))

        return SceneConfig(
            scene_type=scene_type,
            n_rows=n_rows,
            row_spacing=row_spacing,
            point_spacing=point_spacing,
            orientation_deg=orientation_deg,
            field_length=field_length,
            noise_sigma=noise_sigma,
            gap_fraction=gap_fraction,
            gap_max_length=gap_max_length,
            outlier_fraction=outlier_fraction,
            curvature=curvature,
            seed=child_seed,
        )

    # ------------------------------------------------------------------
    # Construção da cena
    # ------------------------------------------------------------------

    def _build_scene(self, config: SceneConfig) -> SyntheticScene:
        rng = np.random.default_rng(config.seed)

        if config.scene_type == "parallel_straight":
            lines_xy = _make_parallel_straight(config)
        elif config.scene_type == "parallel_curved":
            lines_xy = _make_parallel_curved(config)
        else:
            lines_xy = _make_concentric_arcs(config)

        # Amostrar pontos ao longo das linhas de ground truth
        raw_points = _sample_points_from_lines(lines_xy, config, rng)

        # Adicionar outliers (pontos aleatórios no bounding box do campo)
        if config.outlier_fraction > 0:
            raw_points = _add_outliers(raw_points, lines_xy, config, rng)

        return SyntheticScene(points_xy=raw_points, lines_xy=lines_xy, config=config, rng=rng)


# ---------------------------------------------------------------------------
# Geradores de linhas de ground truth
# ---------------------------------------------------------------------------

def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Matriz de rotação 2D."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _make_parallel_straight(config: SceneConfig) -> list[np.ndarray]:
    """Cria fileiras paralelas retas.

    As linhas são geradas no sistema local (x = avança na fileira,
    y = deslocamento transversal) e depois rotacionadas.
    """
    R = _rotation_matrix(config.orientation_deg)
    lines: list[np.ndarray] = []

    # Gerar pontos densamente ao longo do eixo x local
    xs = np.arange(0.0, config.field_length, config.point_spacing / 4)

    for row_idx in range(config.n_rows):
        y_offset = row_idx * config.row_spacing
        local_pts = np.column_stack([xs, np.full_like(xs, y_offset)])
        # Rotacionar e centralizar ao redor da origem
        world_pts = local_pts @ R.T
        lines.append(world_pts)

    return lines


def _make_parallel_curved(config: SceneConfig) -> list[np.ndarray]:
    """Cria fileiras paralelas com curvatura polinomial compartilhada.

    Cada fileira tem o mesmo perfil de curvatura, apenas deslocada
    transversalmente.
    """
    R = _rotation_matrix(config.orientation_deg)
    lines: list[np.ndarray] = []
    xs = np.arange(-config.field_length / 2, config.field_length / 2, config.point_spacing / 4)

    for row_idx in range(config.n_rows):
        # Curvatura: y = row_offset + k * x²
        y_base = row_idx * config.row_spacing + config.curvature * xs**2
        local_pts = np.column_stack([xs, y_base])
        world_pts = local_pts @ R.T
        lines.append(world_pts)

    return lines


def _make_concentric_arcs(config: SceneConfig) -> list[np.ndarray]:
    """Cria arcos concêntricos (tipo pivô central).

    O centro está em (0, 0). Cada fileira é um arco com raio crescente.
    O raio inicial é randomizado para evitar que todos os cenários
    comecem no mesmo ponto.
    """
    lines: list[np.ndarray] = []
    # Raio inicial e extensão angular
    r0 = config.row_spacing * config.n_rows * 0.3
    arc_angle = np.radians(config.orientation_deg % 180 + 60)  # 60 a 240 graus

    for row_idx in range(config.n_rows):
        radius = r0 + row_idx * config.row_spacing
        # Número de pontos proporcional à circunferência
        arc_length = arc_angle * radius
        n_pts = max(4, int(arc_length / (config.point_spacing / 4)))
        angles = np.linspace(0.0, arc_angle, n_pts)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        lines.append(np.column_stack([xs, ys]))

    return lines


# ---------------------------------------------------------------------------
# Amostragem de pontos a partir das linhas
# ---------------------------------------------------------------------------

def _sample_points_from_lines(
    lines: list[np.ndarray],
    config: SceneConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Amostra pontos ao longo das linhas com ruído, gaps e densidade realista."""
    all_points: list[np.ndarray] = []

    for line_pts in lines:
        # Interpolar ao longo da linha pelo espaçamento real dos pontos
        sampled = _resample_line(line_pts, config.point_spacing)
        if len(sampled) < 2:
            all_points.append(sampled)
            continue

        # Adicionar ruído posicional gaussiano
        if config.noise_sigma > 0:
            noise = rng.normal(0.0, config.noise_sigma, size=sampled.shape)
            sampled = sampled + noise

        # Introduzir gaps: remover trechos contíguos de pontos
        sampled = _apply_gaps(sampled, config, rng)

        if len(sampled) > 0:
            all_points.append(sampled)

    if not all_points:
        return np.empty((0, 2))

    return np.vstack(all_points)


def _resample_line(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Reamostrar uma polilinha densa com espaçamento uniforme aproximado."""
    if len(pts) < 2:
        return pts

    # Distâncias acumuladas ao longo da linha
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < spacing:
        return pts[[0]]

    # Posições desejadas ao longo da linha
    target_distances = np.arange(0.0, total_length, spacing)

    # Interpolação linear
    xs = np.interp(target_distances, cumulative, pts[:, 0])
    ys = np.interp(target_distances, cumulative, pts[:, 1])
    return np.column_stack([xs, ys])


def _apply_gaps(pts: np.ndarray, config: SceneConfig, rng: np.random.Generator) -> np.ndarray:
    """Remove trechos contíguos de pontos para simular gaps no campo."""
    if config.gap_fraction <= 0 or len(pts) < 3:
        return pts

    mask = np.ones(len(pts), dtype=bool)
    n_gaps = max(1, int(config.gap_fraction * len(pts) / max(1, config.gap_max_length / config.point_spacing)))

    for _ in range(n_gaps):
        if rng.random() > config.gap_fraction * 3:
            continue
        start_idx = int(rng.integers(0, len(pts)))
        gap_pts = max(1, int(config.gap_max_length / config.point_spacing * rng.random()))
        end_idx = min(len(pts), start_idx + gap_pts)
        mask[start_idx:end_idx] = False

    return pts[mask]


def _add_outliers(
    points: np.ndarray,
    lines: list[np.ndarray],
    config: SceneConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Adiciona pontos aleatórios no bounding box do campo (outliers/ruído)."""
    if len(points) == 0:
        return points

    n_outliers = int(len(points) * config.outlier_fraction)
    if n_outliers == 0:
        return points

    # Bounding box de todos os pontos das linhas
    all_line_pts = np.vstack(lines)
    min_xy = all_line_pts.min(axis=0)
    max_xy = all_line_pts.max(axis=0)

    outlier_xs = rng.uniform(min_xy[0], max_xy[0], size=n_outliers)
    outlier_ys = rng.uniform(min_xy[1], max_xy[1], size=n_outliers)
    outliers = np.column_stack([outlier_xs, outlier_ys])

    return np.vstack([points, outliers])
