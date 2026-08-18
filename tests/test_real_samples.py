from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.ops import unary_union

import main

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


@dataclass(frozen=True)
class ExpectedSample:
    input_name: str
    line_count: int
    supported_point_count: int
    maximum_distance_in_row_spacings: float


# Estes valores formam o contrato de aceitação dos arquivos reais versionados.
# A cobertura é estrutural: pontos sem suporte suficiente não devem criar fileiras.
EXPECTED = {
    1: ExpectedSample("amostra1.gpkg", 19, 291, 0.16),
    # A reconstrução polar da amostra 4 preserva os pontos como vértices.
    4: ExpectedSample("amostra4.geojson", 22, 349, 0.0),
    5: ExpectedSample("amostra5.geojson", 27, 1090, 0.15),
}


@dataclass
class ProcessedSample:
    points: gpd.GeoDataFrame
    metric_points: gpd.GeoDataFrame
    result: gpd.GeoDataFrame
    metric_result: gpd.GeoDataFrame
    row_spacing: float
    distances: np.ndarray


def _process_sample(sample: ExpectedSample) -> ProcessedSample:
    points = main.read_points(DATASET / sample.input_name)
    metric_points = main._working_copy(points)
    xy = np.column_stack((metric_points.geometry.x, metric_points.geometry.y))
    model, _ = main.estimate_spatial_model(xy, cKDTree(xy))

    result = main.generate_lines(points)
    metric_result = result.to_crs(metric_points.crs)
    linework = unary_union(metric_result.geometry.tolist())
    distances = np.fromiter(
        (geometry.distance(linework) for geometry in metric_points.geometry),
        dtype=float,
        count=len(metric_points),
    )
    return ProcessedSample(
        points=points,
        metric_points=metric_points,
        result=result,
        metric_result=metric_result,
        row_spacing=model.row_spacing,
        distances=distances,
    )


class RealSamplesAcceptanceTests(unittest.TestCase):
    """Regressão determinística para os três cenários reais problemáticos."""

    samples: dict[int, ProcessedSample]

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = {
            sample_id: _process_sample(expected)
            for sample_id, expected in EXPECTED.items()
        }

    def test_expected_line_and_supported_point_counts(self) -> None:
        for sample_id, expected in EXPECTED.items():
            with self.subTest(sample=sample_id):
                result = self.samples[sample_id].result
                self.assertEqual(len(result), expected.line_count)
                self.assertEqual(
                    int(result["n_pontos"].sum()),
                    expected.supported_point_count,
                )

    def test_generated_rows_do_not_intersect(self) -> None:
        for sample_id, processed in self.samples.items():
            geometries = processed.metric_result.geometry.reset_index(drop=True)
            with self.subTest(sample=sample_id):
                for first_index in range(len(geometries)):
                    for second_index in range(first_index + 1, len(geometries)):
                        self.assertFalse(
                            geometries.iloc[first_index].intersects(
                                geometries.iloc[second_index]
                            ),
                            msg=(
                                f"Amostra {sample_id}: linhas "
                                f"{first_index + 1} e {second_index + 1} se intersectam."
                            ),
                        )

    def test_supported_points_remain_close_to_rows(self) -> None:
        for sample_id, expected in EXPECTED.items():
            processed = self.samples[sample_id]
            # A saída não persiste o id de cada ponto associado. Selecionar as
            # menores distâncias reproduz, de forma independente, a cobertura
            # estrutural esperada e deixa os pontos rejeitados fora da métrica.
            supported_distances = np.sort(processed.distances)[
                : expected.supported_point_count
            ]
            distance_limit = (
                expected.maximum_distance_in_row_spacings * processed.row_spacing
            )
            with self.subTest(sample=sample_id):
                # Folga apenas para a ida e volta WGS84 -> UTM.
                numerical_tolerance = 1e-4
                self.assertLessEqual(
                    float(supported_distances[-1]),
                    distance_limit + numerical_tolerance,
                    msg=(
                        f"Amostra {sample_id}: ponto estrutural a "
                        f"{supported_distances[-1]:.3f} m da fileira; "
                        f"limite={distance_limit:.3f} m."
                    ),
                )

    def test_all_replantio_points_are_preserved_in_sample_5(self) -> None:
        processed = self.samples[5]
        self.assertIn("CLASSE", processed.points.columns)
        replantio = processed.points["CLASSE"].eq("REPLANTIO").to_numpy()
        self.assertEqual(int(np.count_nonzero(replantio)), 20)

        distance_limit = EXPECTED[5].maximum_distance_in_row_spacings * (
            processed.row_spacing
        )
        self.assertTrue(
            np.all(processed.distances[replantio] <= distance_limit + 1e-6),
            msg="Os 20 pontos REPLANTIO devem permanecer associados às fileiras.",
        )


if __name__ == "__main__":
    unittest.main()
