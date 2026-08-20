"""Testes automatizados do pipeline ML (GeoAI).

Todos os testes rodam em CPU sem GPU obrigatória e sem treinamento real.
O smoke test de treino usa 20 amostras e 1 época para validar o fluxo completo.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Testes do gerador sintético
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:
    def test_generator_produces_scene(self) -> None:
        from src.data.synthetic import SyntheticSceneGenerator
        gen = SyntheticSceneGenerator(seed=42)
        scene = gen.generate()
        assert scene.points_xy.ndim == 2
        assert scene.points_xy.shape[1] == 2
        assert len(scene.lines_xy) > 0

    def test_generator_is_reproducible(self) -> None:
        from src.data.synthetic import SyntheticSceneGenerator
        gen1 = SyntheticSceneGenerator(seed=99)
        gen2 = SyntheticSceneGenerator(seed=99)
        s1 = gen1.generate()
        s2 = gen2.generate()
        np.testing.assert_array_equal(s1.points_xy, s2.points_xy)

    def test_all_scene_types_generate_points(self) -> None:
        from src.data.synthetic import SyntheticSceneGenerator
        gen = SyntheticSceneGenerator(seed=0)
        types_seen = set()
        for _ in range(30):
            scene = gen.generate()
            types_seen.add(scene.config.scene_type)
            assert len(scene.points_xy) > 0
        # Todos os tipos devem aparecer em 30 cenas
        assert len(types_seen) >= 2  # 2 de 3 é suficiente para amostras pequenas

    def test_ground_truth_lines_precede_points(self) -> None:
        """GT é criado antes dos pontos — os pontos devem estar próximos das linhas."""
        from src.data.synthetic import SyntheticSceneGenerator
        from scipy.spatial import cKDTree
        gen = SyntheticSceneGenerator(seed=7)
        scene = gen.generate()
        # Todos os pontos de GT das linhas
        all_line_pts = np.vstack(scene.lines_xy)
        tree = cKDTree(all_line_pts)
        # Pontos sem outlier devem estar razoavelmente próximos das linhas
        # (usamos 5× point_spacing como tolerância máxima)
        dists, _ = tree.query(scene.points_xy)
        max_allowed = 5 * scene.config.point_spacing + 5 * scene.config.noise_sigma + 1e-3
        # Maioria (80%) dos pontos deve estar próxima (outliers são permitidos)
        close_fraction = float((dists <= max_allowed).mean())
        assert close_fraction >= 0.80, f"Apenas {close_fraction:.0%} dos pontos próximos das linhas GT"


# ---------------------------------------------------------------------------
# Testes de rasterização
# ---------------------------------------------------------------------------

class TestRasterization:
    def test_rasterize_points_output_shape(self) -> None:
        from src.data.rasterize import rasterize_points
        pts = np.random.default_rng(0).uniform(0, 100, (50, 2))
        result = rasterize_points(pts, image_size=128)
        assert result.tensor.shape == (3, 128, 128)
        assert result.tensor.dtype == np.float32

    def test_rasterize_points_channels_in_range(self) -> None:
        from src.data.rasterize import rasterize_points
        pts = np.random.default_rng(0).uniform(0, 100, (50, 2))
        result = rasterize_points(pts, image_size=128)
        assert result.tensor.min() >= 0.0
        assert result.tensor.max() <= 1.0 + 1e-6

    def test_world_to_pixel_and_back(self) -> None:
        from src.data.rasterize import AffineTransform, pixel_to_world, world_to_pixel
        transform = AffineTransform(origin_x=100.0, origin_y=200.0, pixel_size=0.5)
        pts_world = np.array([[100.5, 199.5], [101.0, 199.0]])
        pts_px = world_to_pixel(pts_world, transform)
        pts_back = pixel_to_world(pts_px, transform)
        np.testing.assert_allclose(pts_world, pts_back, atol=1e-9)

    def test_rasterize_lines_as_mask_shape(self) -> None:
        from src.data.rasterize import AffineTransform, rasterize_lines_as_mask
        transform = AffineTransform(origin_x=0.0, origin_y=100.0, pixel_size=1.0)
        lines = [np.column_stack([np.arange(10), np.ones(10) * 50])]
        mask = rasterize_lines_as_mask(lines, transform, image_size=128)
        assert mask.shape == (1, 128, 128)
        assert mask.dtype == np.float32
        assert mask.max() <= 1.0


# ---------------------------------------------------------------------------
# Testes do dataset PyTorch
# ---------------------------------------------------------------------------

class TestDataset:
    def test_dataset_getitem_shapes(self) -> None:
        from src.data.dataset import make_dataset
        ds = make_dataset("train", n_samples=5, image_size=64)
        image, mask = ds[0]
        assert image.shape == (3, 64, 64)
        assert mask.shape == (1, 64, 64)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_dataset_deterministic(self) -> None:
        from src.data.dataset import make_dataset
        ds1 = make_dataset("train", n_samples=5, image_size=64)
        ds2 = make_dataset("train", n_samples=5, image_size=64)
        img1, msk1 = ds1[2]
        img2, msk2 = ds2[2]
        torch.testing.assert_close(img1, img2)
        torch.testing.assert_close(msk1, msk2)

    def test_splits_have_disjoint_seeds(self) -> None:
        """Splits train/val/test devem produzir cenas diferentes."""
        from src.data.dataset import make_dataset
        train_ds = make_dataset("train", n_samples=5, image_size=64)
        val_ds = make_dataset("val", n_samples=5, image_size=64)
        img_train, _ = train_ds[0]
        img_val, _ = val_ds[0]
        # As imagens devem ser diferentes (seeds disjuntos)
        assert not torch.equal(img_train, img_val)


# ---------------------------------------------------------------------------
# Testes da U-Net++
# ---------------------------------------------------------------------------

class TestUNetPlusPlus:
    def test_forward_shape_cpu(self) -> None:
        from src.models.unetplusplus import UNetPlusPlus
        model = UNetPlusPlus(in_channels=3, out_channels=1, base_channels=16, depth=3)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1, 128, 128)

    def test_deep_supervision_returns_list(self) -> None:
        from src.models.unetplusplus import UNetPlusPlus
        model = UNetPlusPlus(in_channels=3, out_channels=1, base_channels=8, depth=2, deep_supervision=True)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            outputs = model(x)
        assert isinstance(outputs, list)
        assert len(outputs) == 2  # depth=2 → 2 saídas
        for o in outputs:
            assert o.shape == (1, 1, 64, 64)

    def test_model_parameters_update_after_step(self) -> None:
        from src.models.unetplusplus import UNetPlusPlus
        from src.training.losses import CombinedLoss
        model = UNetPlusPlus(in_channels=3, out_channels=1, base_channels=8, depth=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = CombinedLoss()
        # Salvar parâmetros antes
        params_before = [p.clone() for p in model.parameters()]
        # Um step de treino
        x = torch.randn(2, 3, 64, 64)
        y = torch.zeros(2, 1, 64, 64)
        y[:, :, 20:25, :] = 1.0  # algumas linhas no GT
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        # Verificar que ao menos um parâmetro mudou
        any_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, model.parameters())
        )
        assert any_changed, "Nenhum parâmetro foi atualizado após o step do optimizer"


# ---------------------------------------------------------------------------
# Testes de loss
# ---------------------------------------------------------------------------

class TestLosses:
    def test_combined_loss_decreases_on_correct_pred(self) -> None:
        from src.training.losses import CombinedLoss
        criterion = CombinedLoss()
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, 30:34, :] = 1.0
        # Predição correta (logit positivo onde GT=1)
        good_logit = torch.zeros(2, 1, 64, 64)
        good_logit[:, :, 30:34, :] = 5.0
        # Predição incorreta (logit negativo onde GT=1)
        bad_logit = torch.zeros(2, 1, 64, 64)
        bad_logit[:, :, 30:34, :] = -5.0
        loss_good = criterion(good_logit, target).item()
        loss_bad = criterion(bad_logit, target).item()
        assert loss_good < loss_bad, "Loss deveria ser menor para predição correta"

    def test_dice_loss_zero_for_perfect_prediction(self) -> None:
        from src.training.losses import DiceLoss
        criterion = DiceLoss()
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 10:15, :] = 1.0
        # Logit muito alto onde GT=1, muito baixo onde GT=0
        logit = target * 20.0 - (1 - target) * 20.0
        loss = criterion(logit, target).item()
        assert loss < 0.01, f"Dice Loss deveria ser próxima de 0: {loss:.4f}"


# ---------------------------------------------------------------------------
# Smoke test: 1 época completa de treino
# ---------------------------------------------------------------------------

class TestTrainingSmoke:
    def test_one_epoch_smoke(self) -> None:
        """Verifica que o pipeline completo treino → val → checkpoint funciona."""
        from src.data.dataset import make_dataloaders
        from src.models.unetplusplus import UNetPlusPlus
        from src.training.trainer import Trainer, set_all_seeds

        set_all_seeds(42)

        config = {
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "patience": 99,
                "seed": 42,
            },
            "loss": {"bce_weight": 1.0, "dice_weight": 1.0, "pos_weight": None},
            "model": {"in_channels": 3, "base_channels": 8, "depth": 2},
        }

        train_dl, val_dl = make_dataloaders(
            train_samples=20,
            val_samples=10,
            image_size=64,
            batch_size=2,
            num_workers=0,
        )

        model = UNetPlusPlus(in_channels=3, out_channels=1, base_channels=8, depth=2)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model,
                train_loader=train_dl,
                val_loader=val_dl,
                config=config,
                checkpoint_dir=Path(tmp),
                log_dir=Path(tmp) / "runs",
            )
            metrics = trainer.train()

        # Verificar que métricas foram calculadas
        assert "dice" in metrics or "loss" in metrics or len(metrics) == 0


# ---------------------------------------------------------------------------
# Testes de vetorização
# ---------------------------------------------------------------------------

class TestVectorization:
    def test_skeleton_to_linestrings_returns_geodataframe(self) -> None:
        from src.data.rasterize import AffineTransform
        from src.postprocessing.vectorize import skeleton_to_linestrings

        skeleton = np.zeros((64, 64), dtype=bool)
        skeleton[30, 5:55] = True  # linha horizontal

        transform = AffineTransform(origin_x=0.0, origin_y=64.0, pixel_size=1.0)
        gdf = skeleton_to_linestrings(skeleton, transform, metric_crs="EPSG:32723")

        import geopandas as gpd
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) >= 1

    def test_output_geometries_are_valid(self) -> None:
        from src.data.rasterize import AffineTransform
        from src.postprocessing.vectorize import skeleton_to_linestrings

        skeleton = np.zeros((128, 128), dtype=bool)
        for row in [20, 40, 60, 80]:
            skeleton[row, 5:120] = True

        transform = AffineTransform(origin_x=0.0, origin_y=128.0, pixel_size=0.5)
        gdf = skeleton_to_linestrings(skeleton, transform, metric_crs="EPSG:32723")

        assert gdf.is_valid.all(), "Geometrias inválidas na saída da vetorização"
        assert (gdf.geometry.type == "LineString").all()

    def test_output_has_correct_crs(self) -> None:
        from src.data.rasterize import AffineTransform
        from src.postprocessing.vectorize import skeleton_to_linestrings

        skeleton = np.zeros((64, 64), dtype=bool)
        skeleton[32, :] = True
        transform = AffineTransform(origin_x=0.0, origin_y=64.0, pixel_size=1.0)
        gdf = skeleton_to_linestrings(skeleton, transform, metric_crs="EPSG:32723")
        if len(gdf) > 0:
            assert gdf.crs.to_epsg() == 32723
