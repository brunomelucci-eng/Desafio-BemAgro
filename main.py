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
from scipy.signal import savgol_filter
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
    positive_rows = np.asarray(row_spacing_candidates)
    positive_rows = positive_rows[positive_rows > max(point_spacing * 0.02, 1e-6)]
    row_spacing = (
        float(np.percentile(positive_rows, 25)) if len(positive_rows) else point_spacing
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


def _interval_overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _merge_components_into_rows(
    components: list[list[int]],
    along: np.ndarray,
    straightened: np.ndarray,
    model: SpatialModel,
) -> list[list[int]]:
    """Agrupa fragmentos colineares sem juntar fileiras que coexistem no mesmo trecho."""

    descriptors = []
    for component in components:
        descriptors.append(
            {
                "nodes": component,
                "center": float(np.median(straightened[component])),
                "interval": (
                    float(np.min(along[component])),
                    float(np.max(along[component])),
                ),
            }
        )

    merge_tolerance = max(
        0.35,
        min(0.45 * model.row_spacing, 0.22 * model.point_spacing),
    )
    maximum_overlap = 2.0 * model.point_spacing
    maximum_gap = min(float(np.ptp(along)) * 0.55, model.point_spacing * 20.0)
    groups: list[list[int]] = []

    def group_line(component_indexes: Iterable[int]) -> LineString | None:
        nodes = sorted(
            {
                node
                for index in component_indexes
                for node in descriptors[index]["nodes"]
            },
            key=lambda node: along[node],
        )
        if len(nodes) < 2:
            return None
        # A transformação (avanço, transversal-retificada) preserva a
        # topologia e torna cruzamentos entre fileiras mais fáceis de detectar.
        return LineString([(along[node], straightened[node]) for node in nodes])

    # Componentes longos estabelecem primeiro as fileiras. Fragmentos e pontos
    # isolados só são anexados depois, o que reduz uniões entre vizinhas.
    order = sorted(
        range(len(descriptors)),
        key=lambda idx: (-len(descriptors[idx]["nodes"]), descriptors[idx]["center"]),
    )
    for component_index in order:
        descriptor = descriptors[component_index]
        choices: list[tuple[float, int]] = []
        for group_index, group in enumerate(groups):
            group_centers = [descriptors[idx]["center"] for idx in group]
            center_distance = abs(
                descriptor["center"] - float(np.median(group_centers))
            )
            if center_distance > merge_tolerance:
                continue

            compatible = True
            nearest_gap = float("inf")
            for existing_index in group:
                existing = descriptors[existing_index]
                overlap = _interval_overlap(
                    descriptor["interval"], existing["interval"]
                )
                if overlap > maximum_overlap:
                    compatible = False
                    break
                left_gap = descriptor["interval"][0] - existing["interval"][1]
                right_gap = existing["interval"][0] - descriptor["interval"][1]
                nearest_gap = min(nearest_gap, max(left_gap, right_gap, 0.0))
            if not compatible or nearest_gap > maximum_gap:
                continue

            tentative = group_line([*group, component_index])
            intersects_other_row = False
            if tentative is not None:
                for other_group_index, other_group in enumerate(groups):
                    if other_group_index == group_index:
                        continue
                    other_line = group_line(other_group)
                    if other_line is not None and tentative.intersects(other_line):
                        intersects_other_row = True
                        break
            if not intersects_other_row:
                choices.append((center_distance, group_index))

        if choices:
            _, selected_group = min(choices)
            groups[selected_group].append(component_index)
        else:
            groups.append([component_index])

    rows: list[list[int]] = []
    for group in groups:
        nodes = sorted(
            {
                node
                for component_index in group
                for node in descriptors[component_index]["nodes"]
            },
            key=lambda node: along[node],
        )
        if len(nodes) >= 3:
            rows.append(nodes)
    rows.sort(key=lambda nodes: float(np.median(straightened[nodes])))
    return rows


def _smooth_row(row_xy: np.ndarray) -> np.ndarray:
    """Reduz ruído ponto a ponto sem apagar curvaturas de escala maior."""

    if len(row_xy) < 5:
        return row_xy.copy()
    maximum_window = min(9, len(row_xy))
    window = maximum_window if maximum_window % 2 else maximum_window - 1
    if window < 5:
        return row_xy.copy()
    smoothed = np.column_stack(
        [
            savgol_filter(row_xy[:, axis], window_length=window, polyorder=2)
            for axis in range(2)
        ]
    )
    # Preservar os extremos evita encurtar artificialmente a geometria.
    smoothed[0] = row_xy[0]
    smoothed[-1] = row_xy[-1]
    return smoothed


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
    components, edges = _build_initial_components(xy, tree, model)

    along = xy @ model.direction
    transverse = xy @ model.normal
    straightened = _shared_curve_residual(along, transverse, edges)
    rows = _merge_components_into_rows(components, along, straightened, model)
    if not rows:
        raise ProcessingError(
            "Nenhuma linha com pelo menos três pontos foi identificada."
        )

    records: list[dict[str, object]] = []
    geometries: list[LineString] = []
    for line_id, node_ids in enumerate(rows, start=1):
        ordered_xy = xy[node_ids]
        smoothed_xy = _smooth_row(ordered_xy)
        line = LineString(smoothed_xy)
        length, classification, relative_deviation = _line_metrics(line, smoothed_xy)
        geometries.append(line)
        records.append(
            {
                "id_linha": line_id,
                "comprimento_m": round(length, 3),
                "classificacao": classification,
                "n_pontos": len(node_ids),
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
