"""Gera linhas de tendência a partir de pontos georreferenciados.

O processamento é feito em um CRS UTM local para que distâncias, suavização e
classificação usem metros. A saída é sempre convertida para EPSG:4326.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from shapely.geometry import LineString

SUPPORTED_EXTENSIONS = {".geojson", ".json", ".gpkg", ".shp", ".kml"}


class ProcessingError(RuntimeError):
    """Erro de entrada ou de processamento que pode ser apresentado ao usuário."""


@dataclass(frozen=True)
class SpatialModel:
    """Parâmetros espaciais estimados diretamente do conjunto de pontos."""

    direction: np.ndarray
    normal: np.ndarray
    angle_degrees: float
    angle_tolerance: float
    point_spacing: float
    row_spacing: float


@dataclass(frozen=True)
class ParallelFamily:
    """Fileiras associadas a faixas paralelas robustamente ajustadas."""

    rows: tuple[tuple[int, ...], ...]
    levels: tuple[float, ...]
    direction: np.ndarray
    normal: np.ndarray
    origin: np.ndarray
    spacing: float


@dataclass(frozen=True)
class ConcentricFamily:
    """Fileiras associadas a anéis com um centro comum."""

    rows: tuple[tuple[int, ...], ...]
    center: np.ndarray
    spacing: float


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera linhas de tendência, em GeoJSON/WGS84, a partir de pontos "
            "georreferenciados."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Caminho do arquivo de pontos (.shp, .kml, .geojson ou .gpkg).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Caminho do arquivo GeoJSON que será criado.",
    )
    return parser.parse_args(argv)


def read_points(input_path: Path) -> gpd.GeoDataFrame:
    """Lê e valida uma camada georreferenciada formada somente por pontos."""

    if not input_path.exists():
        raise ProcessingError(f"Arquivo de entrada não encontrado: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ProcessingError(f"Formato não suportado. Use um destes: {extensions}.")

    try:
        points = gpd.read_file(input_path)
    except Exception as exc:  # pyogrio/GDAL fornece a mensagem específica do driver
        raise ProcessingError(f"Não foi possível ler {input_path}: {exc}") from exc

    if points.empty:
        raise ProcessingError("O arquivo de entrada não possui geometrias.")
    if points.crs is None:
        raise ProcessingError(
            "O arquivo de entrada não possui CRS definido; não é possível "
            "calcular comprimentos em metros."
        )

    points = points.explode(index_parts=False, ignore_index=True)
    points = points.loc[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if points.empty or not points.geom_type.eq("Point").all():
        found = (
            ", ".join(sorted(points.geom_type.unique()))
            if not points.empty
            else "vazio"
        )
        raise ProcessingError(
            "A entrada deve conter somente Point ou MultiPoint. "
            f"Geometrias encontradas: {found}."
        )

    # Pontos duplicados geram vetores de tamanho zero e não acrescentam tendência.
    points = points.loc[
        ~points.geometry.apply(lambda geom: geom.wkb).duplicated()
    ].copy()
    points.reset_index(drop=True, inplace=True)
    if len(points) < 3:
        raise ProcessingError("São necessários pelo menos três pontos distintos.")
    return points


def _working_copy(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Converte os pontos para a zona UTM estimada a partir de seu centroide."""

    geographic = points.to_crs(4326)
    metric_crs = geographic.estimate_utm_crs()
    if metric_crs is None:
        raise ProcessingError("Não foi possível estimar um CRS métrico local.")
    return geographic.to_crs(metric_crs)


def _circular_distance(first: int, second: int, period: int) -> int:
    delta = (first - second) % period
    return min(delta, period - delta)


def estimate_spatial_model(
    xy: np.ndarray, tree: cKDTree
) -> tuple[SpatialModel, np.ndarray]:
    """Estima orientação das fileiras e espaçamentos sem limiares em graus/metres fixos."""

    neighbor_count = min(31, len(xy))
    _distances, indices = tree.query(xy, k=neighbor_count)
    nearest_vectors = xy[indices[:, 1]] - xy
    nearest_angles = (
        np.degrees(np.arctan2(nearest_vectors[:, 1], nearest_vectors[:, 0])) % 180.0
    )

    # O histograma usa direções sem sinal: 0° e 180° representam a mesma reta.
    histogram, _ = np.histogram(nearest_angles, bins=360, range=(0.0, 180.0))
    smoothed = gaussian_filter1d(histogram.astype(float), sigma=4.0, mode="wrap")
    peak = int(np.argmax(smoothed))
    angle_degrees = (peak + 0.5) / 2.0

    # A largura do pico distingue fileiras retas de famílias de curvas. O limite
    # superior evita incluir a segunda direção de uma malha regular.
    lobe = {peak}
    cursor = peak - 1
    while smoothed[cursor % 360] >= smoothed[peak] * 0.30 and len(lobe) < 360:
        lobe.add(cursor % 360)
        cursor -= 1
    cursor = peak + 1
    while smoothed[cursor % 360] >= smoothed[peak] * 0.30 and len(lobe) < 360:
        lobe.add(cursor % 360)
        cursor += 1
    half_width = max(_circular_distance(index, peak, 360) for index in lobe) * 0.5
    angle_tolerance = float(np.clip(half_width + 8.0, 15.0, 32.0))

    radians = np.radians(angle_degrees)
    direction = np.array([np.cos(radians), np.sin(radians)])
    normal = np.array([-direction[1], direction[0]])

    point_spacing_candidates: list[float] = []
    row_spacing_candidates: list[float] = []
    for point_index in range(len(xy)):
        vectors = xy[indices[point_index, 1:]] - xy[point_index]
        along = vectors @ direction
        lateral = np.abs(vectors @ normal)
        angles = np.degrees(np.arctan2(lateral, np.abs(along)))
        lengths = np.linalg.norm(vectors, axis=1)

        aligned = angles <= angle_tolerance
        if np.any(aligned):
            point_spacing_candidates.append(float(np.min(lengths[aligned])))

        transverse = np.flatnonzero(angles > max(30.0, angle_tolerance + 5.0))
        if len(transverse):
            closest = transverse[np.argmin(lengths[transverse])]
            row_spacing_candidates.append(float(lateral[closest]))

    if not point_spacing_candidates:
        raise ProcessingError(
            "Não foi possível identificar uma tendência entre os pontos."
        )

    point_spacing = float(np.median(point_spacing_candidates))
    # O menor vizinho transversal nem sempre pertence à fileira adjacente. Em
    # áreas com plantas aleatórias na entrelinha (amostra 1), o percentil 25
    # mede aproximadamente meia entrelinha e transforma ruído em novas linhas.
    # O pico modal de pares quase transversais recupera o espaçamento repetido
    # pela família inteira de fileiras.
    modal_candidates: list[float] = []
    for point_index in range(len(xy)):
        neighbor_ids = indices[point_index, 1:]
        vectors = xy[neighbor_ids] - xy[point_index]
        along = np.abs(vectors @ direction)
        lateral = np.abs(vectors @ normal)
        valid = (
            (along <= 0.50 * point_spacing)
            & (lateral >= 0.25 * point_spacing)
            & (lateral <= 2.50 * point_spacing)
        )
        modal_candidates.extend(lateral[valid].tolist())

    row_spacing: float | None = None
    if modal_candidates:
        bin_width = max(0.04 * point_spacing, 1e-6)
        lower = 0.25 * point_spacing
        upper = 2.50 * point_spacing
        bin_edges = np.arange(lower, upper + bin_width, bin_width)
        histogram, bin_edges = np.histogram(modal_candidates, bins=bin_edges)
        density = gaussian_filter1d(histogram.astype(float), sigma=1.5)
        peaks, _ = find_peaks(
            density,
            prominence=max(0.05 * float(np.max(density)), 0.05),
        )
        if len(peaks):
            selected = int(peaks[np.argmax(density[peaks])])
            row_spacing = float(0.5 * (bin_edges[selected] + bin_edges[selected + 1]))

    if row_spacing is None:
        positive_rows = np.asarray(row_spacing_candidates)
        positive_rows = positive_rows[positive_rows > max(point_spacing * 0.02, 1e-6)]
        row_spacing = (
            float(np.median(positive_rows)) if len(positive_rows) else point_spacing
        )

    model = SpatialModel(
        direction=direction,
        normal=normal,
        angle_degrees=angle_degrees,
        angle_tolerance=angle_tolerance,
        point_spacing=max(point_spacing, 1e-6),
        row_spacing=max(row_spacing, point_spacing * 0.20, 1e-6),
    )
    return model, indices


def _local_tangent_seeds(
    xy: np.ndarray,
    tree: cKDTree,
    model: SpatialModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtém tangentes locais curtas sem permitir que um outlier crie uma trilha."""

    along = xy @ model.direction
    seed_indexes: list[int] = []
    tangents: list[np.ndarray] = []
    search_radius = 1.75 * model.point_spacing

    for point_index in range(len(xy)):
        neighbors = np.asarray(
            [
                index
                for index in tree.query_ball_point(xy[point_index], search_radius)
                if index != point_index
            ],
            dtype=int,
        )
        if not len(neighbors):
            continue

        delta_along = along[neighbors] - along[point_index]
        previous = neighbors[delta_along < -0.35 * model.point_spacing]
        following = neighbors[delta_along > 0.35 * model.point_spacing]
        best: tuple[float, int, int, float, float] | None = None

        for start in previous:
            first = xy[point_index] - xy[start]
            first_length = float(np.linalg.norm(first))
            if first_length <= 1e-9:
                continue
            for end in following:
                second = xy[end] - xy[point_index]
                second_length = float(np.linalg.norm(second))
                if second_length <= 1e-9:
                    continue
                cosine = float(
                    np.clip(
                        np.dot(first, second) / (first_length * second_length),
                        -1.0,
                        1.0,
                    )
                )
                turn = float(np.degrees(np.arccos(cosine)))
                step_error = (
                    abs(first_length - model.point_spacing)
                    + abs(second_length - model.point_spacing)
                ) / model.point_spacing
                score = turn / 8.0 + step_error
                candidate = (score, int(start), int(end), turn, step_error)
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None or best[3] > 8.0 or best[4] > 0.50:
            continue
        tangent = xy[best[2]] - xy[best[1]]
        tangent /= np.linalg.norm(tangent)
        if float(tangent @ model.direction) < 0.0:
            tangent = -tangent
        seed_indexes.append(point_index)
        tangents.append(tangent)

    if not seed_indexes:
        return np.empty(0, dtype=int), np.empty((0, 2), dtype=float)
    return np.asarray(seed_indexes, dtype=int), np.asarray(tangents)


def _best_lattice_phase(
    values: np.ndarray,
    spacing: float,
    trim_fraction: float = 0.95,
) -> float:
    """Escolhe a fase da grade periódica ignorando a cauda de outliers."""

    kept_count = max(1, int(np.floor(trim_fraction * len(values))))
    best_score = float("inf")
    best_phase = 0.0
    for phase in np.linspace(0.0, spacing, 256, endpoint=False):
        residuals = np.abs((values - phase + 0.5 * spacing) % spacing - 0.5 * spacing)
        trimmed = np.partition(residuals, kept_count - 1)[:kept_count]
        score = float(np.mean(trimmed))
        if score < best_score:
            best_score = score
            best_phase = float(phase)
    return best_phase


def _concentric_center_candidate(
    xy: np.ndarray,
    seed_indexes: np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray | None:
    """Detecta quando as tangentes descrevem uma família circular confiável."""

    if len(seed_indexes) < 30:
        return None

    origin = np.mean(xy, axis=0)
    seed_points = xy[seed_indexes] - origin
    targets = np.sum(tangents * seed_points, axis=1)
    keep = np.ones(len(seed_indexes), dtype=bool)
    center = np.zeros(2)

    for _ in range(6):
        center = np.linalg.lstsq(tangents[keep], targets[keep], rcond=None)[0]
        errors = tangents @ center - targets
        median = float(np.median(errors[keep]))
        mad = float(np.median(np.abs(errors[keep] - median)))
        if mad <= 1e-9:
            break
        next_keep = np.abs(errors - median) <= 3.0 * 1.4826 * mad
        if np.count_nonzero(next_keep) < 10 or np.array_equal(next_keep, keep):
            break
        keep = next_keep

    singular_values = np.linalg.svd(tangents[keep], compute_uv=False)
    if (
        len(singular_values) < 2
        or singular_values[0] <= 1e-9
        or singular_values[-1] / singular_values[0] < 0.15
    ):
        return None

    radial = seed_points - center
    radial_lengths = np.linalg.norm(radial, axis=1)
    valid_radial = radial_lengths > 1e-9
    if np.count_nonzero(valid_radial) < 30:
        return None
    radial[valid_radial] /= radial_lengths[valid_radial, None]
    perpendicular_error = np.abs(
        np.sum(tangents[valid_radial] * radial[valid_radial], axis=1)
    )
    angular_errors = np.degrees(np.arcsin(np.clip(perpendicular_error, 0.0, 1.0)))
    median_error, percentile_90 = np.percentile(angular_errors, [50, 90])
    if median_error > 5.0 or percentile_90 > 10.0:
        return None
    return origin + center


def _fit_concentric_family(
    xy: np.ndarray,
    initial_center: np.ndarray,
    model: SpatialModel,
) -> ConcentricFamily | None:
    """Ajusta centro, espaçamento e anéis; aceita apenas pontos na banda radial."""

    origin = np.mean(xy, axis=0)
    initial_relative_center = initial_center - origin
    center = initial_relative_center.copy()
    spacing = model.row_spacing
    radii = np.linalg.norm(xy - origin - center, axis=1)
    phase = _best_lattice_phase(radii, spacing)
    labels = np.rint((radii - phase) / spacing).astype(int)

    for _ in range(8):
        previous_labels = labels.copy()
        fixed_labels = labels.copy()

        def residuals(
            parameters: np.ndarray,
            labels_for_fit: np.ndarray = fixed_labels,
        ) -> np.ndarray:
            candidate_center = parameters[:2]
            candidate_spacing = parameters[2]
            candidate_phase = parameters[3]
            candidate_radii = np.linalg.norm(xy - origin - candidate_center, axis=1)
            return candidate_radii - (
                candidate_phase + labels_for_fit * candidate_spacing
            )

        center_margin = 3.0 * model.row_spacing
        result = least_squares(
            residuals,
            np.asarray([center[0], center[1], spacing, phase]),
            loss="soft_l1",
            f_scale=0.25 * model.row_spacing,
            bounds=(
                [
                    initial_relative_center[0] - center_margin,
                    initial_relative_center[1] - center_margin,
                    0.80 * model.row_spacing,
                    -10.0 * model.row_spacing,
                ],
                [
                    initial_relative_center[0] + center_margin,
                    initial_relative_center[1] + center_margin,
                    1.25 * model.row_spacing,
                    10.0 * model.row_spacing,
                ],
            ),
        )
        center = result.x[:2]
        spacing = float(result.x[2])
        phase = float(result.x[3])
        radii = np.linalg.norm(xy - origin - center, axis=1)
        labels = np.rint((radii - phase) / spacing).astype(int)
        if np.array_equal(labels, previous_labels):
            break

    radial_residuals = radii - (phase + labels * spacing)
    accepted = np.abs(radial_residuals) <= 0.45 * spacing
    rows: list[tuple[int, ...]] = []
    for label in sorted(np.unique(labels[accepted])):
        nodes = tuple(np.flatnonzero(accepted & (labels == label)).tolist())
        if len(nodes) >= 3:
            rows.append(nodes)
    assigned = sum(map(len, rows))
    if len(rows) < 2 or assigned < 0.85 * len(xy):
        return None
    return ConcentricFamily(
        rows=tuple(rows),
        center=origin + center,
        spacing=spacing,
    )


def _fit_parallel_family(
    xy: np.ndarray,
    seed_indexes: np.ndarray,
    tangents: np.ndarray,
    model: SpatialModel,
) -> ParallelFamily | None:
    """Ajusta uma grade de faixas paralelas e rejeita pontos entre fileiras."""

    if len(seed_indexes) < 3:
        return None
    origin = np.mean(xy, axis=0)
    centered = xy - origin
    _, eigenvectors = np.linalg.eigh(tangents.T @ tangents)
    initial_direction = eigenvectors[:, -1]
    if float(initial_direction @ model.direction) < 0.0:
        initial_direction = -initial_direction
    initial_angle = float(np.arctan2(initial_direction[1], initial_direction[0]))
    initial_normal = np.asarray([-np.sin(initial_angle), np.cos(initial_angle)])
    transverse = centered @ initial_normal
    spacing = model.row_spacing
    phase = _best_lattice_phase(transverse, spacing)
    labels = np.rint((transverse - phase) / spacing).astype(int)
    angle_delta = 0.0

    for _ in range(8):
        previous_labels = labels.copy()
        fixed_labels = labels.copy()

        def residuals(
            parameters: np.ndarray,
            labels_for_fit: np.ndarray = fixed_labels,
        ) -> np.ndarray:
            angle = initial_angle + parameters[0]
            normal = np.asarray([-np.sin(angle), np.cos(angle)])
            return centered @ normal - (parameters[2] + labels_for_fit * parameters[1])

        result = least_squares(
            residuals,
            np.asarray([angle_delta, spacing, phase]),
            loss="soft_l1",
            f_scale=0.25 * model.row_spacing,
            bounds=(
                [
                    -np.radians(10.0),
                    0.80 * model.row_spacing,
                    -10.0 * model.row_spacing,
                ],
                [
                    np.radians(10.0),
                    1.25 * model.row_spacing,
                    10.0 * model.row_spacing,
                ],
            ),
        )
        angle_delta, spacing, phase = map(float, result.x)
        angle = initial_angle + angle_delta
        normal = np.asarray([-np.sin(angle), np.cos(angle)])
        transverse = centered @ normal
        labels = np.rint((transverse - phase) / spacing).astype(int)
        if np.array_equal(labels, previous_labels):
            break

    direction = np.asarray([np.cos(angle), np.sin(angle)])
    normal = np.asarray([-direction[1], direction[0]])
    residual = transverse - (phase + labels * spacing)
    accepted = np.abs(residual) <= 0.15 * spacing
    rows: list[tuple[int, ...]] = []
    levels: list[float] = []
    for label in sorted(np.unique(labels[accepted])):
        nodes = tuple(np.flatnonzero(accepted & (labels == label)).tolist())
        if len(nodes) >= 3:
            rows.append(nodes)
            levels.append(float(phase + label * spacing))
    assigned = sum(map(len, rows))
    # Famílias com curvatura compartilhada ou ruído difuso seguem para o
    # detector flexível. Esta solução só vence quando a grade paralela explica
    # quase todo o conjunto (caso da amostra 5).
    if len(rows) < 2 or assigned < 0.90 * len(xy):
        return None
    return ParallelFamily(
        rows=tuple(rows),
        levels=tuple(levels),
        direction=direction,
        normal=normal,
        origin=origin,
        spacing=spacing,
    )


def _connected_components(
    adjacency: dict[int, list[int]], count: int
) -> list[list[int]]:
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(count):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _segments_cross(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    """Retorna True somente para uma interseção própria entre dois segmentos."""

    if (
        max(first_start[0], first_end[0]) < min(second_start[0], second_end[0])
        or max(second_start[0], second_end[0]) < min(first_start[0], first_end[0])
        or max(first_start[1], first_end[1]) < min(second_start[1], second_end[1])
        or max(second_start[1], second_end[1]) < min(first_start[1], first_end[1])
    ):
        return False

    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        return float(
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )

    first_side = orientation(first_start, first_end, second_start)
    second_side = orientation(first_start, first_end, second_end)
    third_side = orientation(second_start, second_end, first_start)
    fourth_side = orientation(second_start, second_end, first_end)
    return first_side * second_side < -1e-9 and third_side * fourth_side < -1e-9


def _build_initial_components(
    xy: np.ndarray,
    tree: cKDTree,
    model: SpatialModel,
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Liga pontos com no máximo um sucessor e um antecessor."""

    maximum_distance = 5.0 * model.point_spacing
    candidates: list[tuple[float, int, int]] = []

    for point_index in range(len(xy)):
        neighbor_ids = np.asarray(
            [
                idx
                for idx in tree.query_ball_point(xy[point_index], maximum_distance)
                if idx != point_index
            ],
            dtype=int,
        )
        if not len(neighbor_ids):
            continue

        vectors = xy[neighbor_ids] - xy[point_index]
        along = vectors @ model.direction
        lateral = np.abs(vectors @ model.normal)
        angles = np.degrees(np.arctan2(lateral, np.maximum(along, 1e-9)))
        lengths = np.linalg.norm(vectors, axis=1)
        valid = (along > 0.0) & (angles <= model.angle_tolerance)

        for neighbor, length, angle in zip(
            neighbor_ids[valid], lengths[valid], angles[valid], strict=True
        ):
            score = (length / model.point_spacing) * (
                1.0 + 1.2 * (angle / model.angle_tolerance) ** 2
            )
            if score < 4.25:
                candidates.append((float(score), point_index, int(neighbor)))

    used_successors: set[int] = set()
    used_predecessors: set[int] = set()
    adjacency: dict[int, list[int]] = defaultdict(list)
    edges: list[tuple[int, int]] = []
    for _, start, end in sorted(candidates):
        if start in used_successors or end in used_predecessors:
            continue
        if any(
            start not in (edge_start, edge_end)
            and end not in (edge_start, edge_end)
            and _segments_cross(xy[start], xy[end], xy[edge_start], xy[edge_end])
            for edge_start, edge_end in edges
        ):
            continue
        used_successors.add(start)
        used_predecessors.add(end)
        adjacency[start].append(end)
        adjacency[end].append(start)
        edges.append((start, end))

    return _connected_components(adjacency, len(xy)), edges


def _shared_curve_residual(
    along: np.ndarray,
    transverse: np.ndarray,
    edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Remove a curvatura compartilhada, preservando o afastamento entre fileiras."""

    center = float(np.mean(along))
    scale = max(float(np.ptp(along)), 1.0)
    normalized = (along - center) / scale
    design_rows: list[list[float]] = []
    targets: list[float] = []
    for start, end in edges:
        design_rows.append(
            [
                normalized[end] - normalized[start],
                normalized[end] ** 2 - normalized[start] ** 2,
                normalized[end] ** 3 - normalized[start] ** 3,
            ]
        )
        targets.append(float(transverse[end] - transverse[start]))

    if len(design_rows) < 3:
        return transverse.copy()

    design = np.asarray(design_rows)
    target = np.asarray(targets)
    keep = np.ones(len(target), dtype=bool)
    coefficients = np.zeros(3)
    for _ in range(4):
        coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        errors = target - design @ coefficients
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        if mad <= 1e-9:
            break
        keep = np.abs(errors - median) < 3.5 * 1.4826 * mad
        if np.count_nonzero(keep) < 3:
            break

    shared_curve = (
        coefficients[0] * normalized
        + coefficients[1] * normalized**2
        + coefficients[2] * normalized**3
    )
    return transverse - shared_curve


def _density_rows(
    straightened: np.ndarray,
    row_spacing: float,
) -> list[tuple[list[int], float]]:
    """Localiza cristas densas e descarta cristas esparsas na entrelinha."""

    bin_width = max(row_spacing / 30.0, 1e-6)
    bin_edges = np.arange(
        float(np.min(straightened)) - row_spacing,
        float(np.max(straightened)) + row_spacing + bin_width,
        bin_width,
    )
    histogram, bin_edges = np.histogram(straightened, bins=bin_edges)
    density = gaussian_filter1d(
        histogram.astype(float),
        sigma=0.12 * row_spacing / bin_width,
    )
    if not len(density) or float(np.max(density)) <= 0.0:
        return []
    peaks, _ = find_peaks(
        density,
        distance=max(1, int(0.65 * row_spacing / bin_width)),
        prominence=0.03 * float(np.max(density)),
    )
    if not len(peaks):
        return []

    # Uma fileira real se repete ao longo de vários avanços. Picos com menos
    # de 30% da densidade dominante são fragmentos/outliers (amostra 1), não
    # uma nova hipótese de fileira.
    peaks = peaks[density[peaks] >= 0.30 * float(np.max(density[peaks]))]
    centers = 0.5 * (bin_edges[peaks] + bin_edges[peaks + 1])
    if not len(centers):
        return []

    distances = np.abs(straightened[:, None] - centers[None, :])
    labels = np.argmin(distances, axis=1)
    accepted = distances[np.arange(len(straightened)), labels] <= 0.16 * row_spacing
    rows: list[tuple[list[int], float]] = []
    for label, center in enumerate(centers):
        nodes = np.flatnonzero(accepted & (labels == label)).tolist()
        if len(nodes) >= 3:
            rows.append((nodes, float(center)))
    return rows


def _parallel_line_geometry(
    xy: np.ndarray,
    nodes: Sequence[int],
    level: float,
    family: ParallelFamily,
) -> np.ndarray:
    """Cria a tendência reta central da faixa, inclusive através de lacunas."""

    centered = xy[np.asarray(nodes)] - family.origin
    along = centered @ family.direction
    endpoints = np.asarray([float(np.min(along)), float(np.max(along))])
    return family.origin + endpoints[:, None] * family.direction + level * family.normal


def _polar_line_geometry(
    xy: np.ndarray,
    nodes: Sequence[int],
    center: np.ndarray,
    point_spacing: float,
) -> np.ndarray:
    """Interpola em coordenadas polares para uma lacuna não virar uma corda."""

    node_array = np.asarray(nodes, dtype=int)
    vectors = xy[node_array] - center
    radii = np.linalg.norm(vectors, axis=1)
    raw_angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * np.pi)
    order = np.argsort(raw_angles)
    ordered_angles = raw_angles[order]
    circular_gaps = np.diff(np.r_[ordered_angles, ordered_angles[0] + 2.0 * np.pi])
    first = (int(np.argmax(circular_gaps)) + 1) % len(order)
    order = np.roll(order, -first)
    ordered_angles = np.unwrap(raw_angles[order])
    ordered_radii = radii[order]

    dense_angles: list[float] = []
    dense_radii: list[float] = []
    maximum_step = max(0.75 * point_spacing, 1e-6)
    for index in range(len(order) - 1):
        angle_start = float(ordered_angles[index])
        angle_end = float(ordered_angles[index + 1])
        radius_start = float(ordered_radii[index])
        radius_end = float(ordered_radii[index + 1])
        arc_length = abs(angle_end - angle_start) * 0.5 * (radius_start + radius_end)
        segment_count = max(1, int(np.ceil(arc_length / maximum_step)))
        dense_angles.extend(
            np.linspace(
                angle_start,
                angle_end,
                segment_count,
                endpoint=False,
            ).tolist()
        )
        dense_radii.extend(
            np.linspace(
                radius_start,
                radius_end,
                segment_count,
                endpoint=False,
            ).tolist()
        )
    dense_angles.append(float(ordered_angles[-1]))
    dense_radii.append(float(ordered_radii[-1]))
    angles = np.asarray(dense_angles)
    radii_array = np.asarray(dense_radii)
    return center + np.column_stack(
        (radii_array * np.cos(angles), radii_array * np.sin(angles))
    )


def _shared_curve_line_geometry(
    nodes: Sequence[int],
    row_center: float,
    along: np.ndarray,
    transverse: np.ndarray,
    straightened: np.ndarray,
    model: SpatialModel,
) -> np.ndarray:
    """Interpola uma fileira dentro de sua faixa retificada, sem atalhos."""

    node_array = np.asarray(nodes, dtype=int)
    global_center = float(np.mean(along))
    global_scale = max(float(np.ptp(along)), 1.0)
    normalized = (along - global_center) / global_scale
    shared_degree = min(3, len(along) - 1)
    shared_coefficients = np.polyfit(
        normalized,
        transverse - straightened,
        shared_degree,
    )

    order = node_array[np.argsort(along[node_array])]
    node_along = along[order]
    node_residual = straightened[order]
    dense_along_parts: list[np.ndarray] = []
    dense_residual_parts: list[np.ndarray] = []
    maximum_step = max(0.75 * model.point_spacing, 1e-6)
    for index in range(len(order) - 1):
        delta_along = float(node_along[index + 1] - node_along[index])
        segment_count = max(1, int(np.ceil(abs(delta_along) / maximum_step)))
        dense_along_parts.append(
            np.linspace(
                node_along[index],
                node_along[index + 1],
                segment_count,
                endpoint=False,
            )
        )
        dense_residual_parts.append(
            np.linspace(
                node_residual[index],
                node_residual[index + 1],
                segment_count,
                endpoint=False,
            )
        )
    dense_along = np.concatenate([*dense_along_parts, np.asarray([node_along[-1]])])
    dense_residual = np.concatenate(
        [*dense_residual_parts, np.asarray([node_residual[-1]])]
    )
    dense_normalized = (dense_along - global_center) / global_scale
    shared_curve = np.polyval(shared_coefficients, dense_normalized)
    dense_residual = np.clip(
        dense_residual,
        row_center - 0.16 * model.row_spacing,
        row_center + 0.16 * model.row_spacing,
    )
    dense_transverse = shared_curve + dense_residual
    return (
        dense_along[:, None] * model.direction
        + dense_transverse[:, None] * model.normal
    )


def _line_metrics(line: LineString, row_xy: np.ndarray) -> tuple[float, str, float]:
    """Calcula comprimento e classificação por desvio relativo e mudança angular."""

    length = float(line.length)
    chord_vector = row_xy[-1] - row_xy[0]
    chord = float(np.linalg.norm(chord_vector))
    if chord <= 1e-9 or len(row_xy) < 4:
        return length, "reta", 0.0

    unit = chord_vector / chord
    normal = np.array([-unit[1], unit[0]])
    chord_deviation = np.abs((row_xy - row_xy[0]) @ normal)
    relative_deviation = float(np.max(chord_deviation) / chord)

    segments = np.diff(row_xy, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    segments = segments[segment_lengths > 1e-9]
    if len(segments) >= 2:
        angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
        turning_degrees = float(np.degrees(np.max(angles) - np.min(angles)))
    else:
        turning_degrees = 0.0

    sinuosity = length / chord
    is_curve = (
        (relative_deviation >= 0.020 and turning_degrees >= 8.0)
        or (relative_deviation >= 0.035)
        or (sinuosity >= 1.015 and turning_degrees >= 10.0)
    )
    return length, "curva" if is_curve else "reta", relative_deviation


def generate_lines(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Executa o algoritmo completo e devolve linhas em EPSG:4326."""

    metric_points = _working_copy(points)
    xy = np.column_stack((metric_points.geometry.x, metric_points.geometry.y))
    tree = cKDTree(xy)
    model, _ = estimate_spatial_model(xy, tree)
    seed_indexes, tangents = _local_tangent_seeds(xy, tree, model)

    line_inputs: list[tuple[np.ndarray, int]] = []
    circular_center = _concentric_center_candidate(xy, seed_indexes, tangents)
    concentric = (
        _fit_concentric_family(xy, circular_center, model)
        if circular_center is not None
        else None
    )
    if concentric is not None:
        for nodes in concentric.rows:
            geometry_xy = _polar_line_geometry(
                xy,
                nodes,
                concentric.center,
                model.point_spacing,
            )
            line_inputs.append((geometry_xy, len(nodes)))
    else:
        parallel = _fit_parallel_family(
            xy,
            seed_indexes,
            tangents,
            model,
        )
        if parallel is not None:
            for nodes, level in zip(
                parallel.rows,
                parallel.levels,
                strict=True,
            ):
                geometry_xy = _parallel_line_geometry(
                    xy,
                    nodes,
                    level,
                    parallel,
                )
                line_inputs.append((geometry_xy, len(nodes)))
        else:
            _components, edges = _build_initial_components(xy, tree, model)
            along = xy @ model.direction
            transverse = xy @ model.normal
            straightened = _shared_curve_residual(along, transverse, edges)
            density_rows = _density_rows(straightened, model.row_spacing)
            for nodes, row_center in density_rows:
                geometry_xy = _shared_curve_line_geometry(
                    nodes,
                    row_center,
                    along,
                    transverse,
                    straightened,
                    model,
                )
                line_inputs.append((geometry_xy, len(nodes)))

    if not line_inputs:
        raise ProcessingError(
            "Nenhuma linha com pelo menos três pontos foi identificada."
        )

    records: list[dict[str, object]] = []
    geometries: list[LineString] = []
    for line_id, (geometry_xy, point_count) in enumerate(line_inputs, start=1):
        line = LineString(geometry_xy)
        length, classification, relative_deviation = _line_metrics(line, geometry_xy)
        geometries.append(line)
        records.append(
            {
                "id_linha": line_id,
                "comprimento_m": round(length, 3),
                "classificacao": classification,
                "n_pontos": point_count,
                "desvio_relativo": round(relative_deviation, 5),
            }
        )

    result = gpd.GeoDataFrame(records, geometry=geometries, crs=metric_points.crs)
    return result.to_crs(4326)


def run(input_path: Path, output_path: Path) -> gpd.GeoDataFrame:
    """Lê a entrada, gera linhas e salva o GeoJSON final."""

    if output_path.suffix.lower() not in {".geojson", ".json"}:
        raise ProcessingError("O caminho de saída deve terminar em .geojson ou .json.")
    if input_path.resolve() == output_path.resolve():
        raise ProcessingError("Entrada e saída precisam ser arquivos diferentes.")

    points = read_points(input_path)
    result = generate_lines(points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_path, driver="GeoJSON", index=False)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args.input, args.output)
    except ProcessingError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - fronteira do CLI; evita traceback para o usuário
        print(f"Erro inesperado durante o processamento: {exc}", file=sys.stderr)
        return 1

    counts = result["classificacao"].value_counts().to_dict()
    print(
        f"Concluído: {len(result)} linhas salvas em {args.output} "
        f"(retas={counts.get('reta', 0)}, curvas={counts.get('curva', 0)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
