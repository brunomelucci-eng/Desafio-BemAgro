from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import main

METRIC_CRS = "EPSG:32723"


def straight_rows() -> gpd.GeoDataFrame:
    points = []
    for row in range(4):
        for x_coordinate in np.arange(0.0, 52.0, 2.0):
            if row == 2 and x_coordinate in (20.0, 22.0, 24.0, 26.0):
                continue
            points.append(Point(300_000.0 + x_coordinate, 7_700_000.0 + row * 4.0))
    return gpd.GeoDataFrame(geometry=points, crs=METRIC_CRS)


def curved_rows() -> gpd.GeoDataFrame:
    points = []
    for row in range(4):
        for x_coordinate in np.arange(-30.0, 32.0, 2.0):
            if row == 1 and -4.0 <= x_coordinate <= 4.0:
                continue
            y_coordinate = row * 4.0 + 0.004 * x_coordinate**2
            points.append(Point(300_000.0 + x_coordinate, 7_700_000.0 + y_coordinate))
    return gpd.GeoDataFrame(geometry=points, crs=METRIC_CRS)


def straight_rows_with_sparse_interrow_points() -> gpd.GeoDataFrame:
    """Cria quatro fileiras reais e uma sequência esparsa na entrelinha."""

    points = []
    for row in range(4):
        for x_coordinate in np.arange(0.0, 52.0, 2.0):
            if row == 2 and x_coordinate in (
                20.0,
                22.0,
                24.0,
                26.0,
                28.0,
                30.0,
            ):
                continue
            points.append(Point(300_000.0 + x_coordinate, 7_700_000.0 + row * 4.0))

    # Estes pontos são colineares, mas têm apenas 1/4 da densidade das fileiras
    # verdadeiras e ficam exatamente entre duas delas. Não devem originar linha.
    for x_coordinate in (7.0, 15.0, 23.0, 31.0, 39.0, 47.0):
        points.append(Point(300_000.0 + x_coordinate, 7_700_006.0))

    return gpd.GeoDataFrame(geometry=points, crs=METRIC_CRS)


class GenerateLinesTests(unittest.TestCase):
    def test_straight_rows_and_gap_are_reconstructed(self) -> None:
        result = main.generate_lines(straight_rows())

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["classificacao"]), {"reta"})
        self.assertEqual(sorted(result["n_pontos"]), [22, 26, 26, 26])
        self.assertTrue((result["comprimento_m"] > 49.0).all())
        self.assertEqual(result.crs.to_epsg(), 4326)

    def test_parallel_curves_are_classified_and_gap_is_filled(self) -> None:
        result = main.generate_lines(curved_rows())

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["classificacao"]), {"curva"})
        self.assertEqual(sorted(result["n_pontos"]), [26, 31, 31, 31])

    def test_generated_lines_do_not_cross(self) -> None:
        result = main.generate_lines(curved_rows()).to_crs(METRIC_CRS)
        for first in range(len(result)):
            for second in range(first + 1, len(result)):
                self.assertFalse(
                    result.geometry.iloc[first].intersects(result.geometry.iloc[second])
                )

    def test_sparse_interrow_points_are_ignored_and_real_gap_is_joined(self) -> None:
        result = main.generate_lines(straight_rows_with_sparse_interrow_points())

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["classificacao"]), {"reta"})
        self.assertEqual(sorted(result["n_pontos"]), [20, 26, 26, 26])

        row_with_gap = result.loc[result["n_pontos"] == 20]
        self.assertEqual(len(row_with_gap), 1)
        self.assertGreater(float(row_with_gap.iloc[0]["comprimento_m"]), 49.0)

    def test_shuffled_input_preserves_line_and_point_counts(self) -> None:
        points = curved_rows()
        permutation = np.random.default_rng(20260817).permutation(len(points))
        shuffled = points.iloc[permutation].reset_index(drop=True)

        original_result = main.generate_lines(points)
        shuffled_result = main.generate_lines(shuffled)

        self.assertEqual(len(original_result), len(shuffled_result))
        self.assertEqual(
            original_result["classificacao"].value_counts().to_dict(),
            shuffled_result["classificacao"].value_counts().to_dict(),
        )
        self.assertEqual(
            sorted(original_result["n_pontos"].tolist()),
            sorted(shuffled_result["n_pontos"].tolist()),
        )


class InputOutputTests(unittest.TestCase):
    def test_input_without_crs_is_rejected(self) -> None:
        points = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0), Point(2, 0)])
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "sem_crs.geojson"
            input_path.touch()
            with (
                mock.patch("main.gpd.read_file", return_value=points),
                self.assertRaisesRegex(main.ProcessingError, "não possui CRS"),
            ):
                main.read_points(input_path)

    def test_run_writes_expected_schema_in_wgs84(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "pontos.geojson"
            output_path = root / "linhas.geojson"
            straight_rows().to_crs(4326).to_file(
                input_path, driver="GeoJSON", index=False
            )

            result = main.run(input_path, output_path)
            saved = gpd.read_file(output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(result.crs.to_epsg(), 4326)
            self.assertEqual(saved.crs.to_epsg(), 4326)
            self.assertEqual(
                list(saved.columns),
                [
                    "id_linha",
                    "comprimento_m",
                    "classificacao",
                    "n_pontos",
                    "desvio_relativo",
                    "geometry",
                ],
            )
            self.assertTrue(saved.is_valid.all())


if __name__ == "__main__":
    unittest.main()
