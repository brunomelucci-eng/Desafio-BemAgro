"""Pipeline de inferência ML completo com tile inference e weighted blending.

Uso rápido
----------
    from src.inference.predictor import MLPredictor
    predictor = MLPredictor("models/best_model.pth")
    result_gdf = predictor.predict(input_path, output_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

from src.data.rasterize import rasterize_points
from src.models.unetplusplus import UNetPlusPlus

logger = logging.getLogger(__name__)


class MLPredictor:
    """Pipeline de inferência geoespacial com U-Net++.

    Fluxo interno:
        read_points → validate_crs → reproject_UTM → rasterize
        → load_model → inference (tile) → sigmoid → skeleton
        → vectorize → classify → reproject_WGS84 → GeoJSON

    Parameters
    ----------
    weights_path: Path para o arquivo .pth do modelo treinado.
    threshold: Limiar de binarização do mapa de probabilidades (default 0.5).
    tile_size: Tamanho dos tiles para inferência (256 ou 512).
    overlap_fraction: Fração de sobreposição entre tiles (default 0.25).
    device: "cpu", "cuda" ou None (detecta automaticamente).
    """

    def __init__(
        self,
        weights_path: Path | str,
        threshold: float = 0.5,
        tile_size: int = 256,
        overlap_fraction: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.threshold = threshold
        self.tile_size = tile_size
        self.overlap_fraction = overlap_fraction

        # Dispositivo automático
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Carregar modelo
        self.model, self.model_config = self._load_model()

    def _load_model(self) -> tuple[UNetPlusPlus, dict]:
        """Carrega checkpoint e reconstrói o modelo."""
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        config = checkpoint.get("config", {})
        model_cfg = config.get("model", {})

        model = UNetPlusPlus(
            in_channels=model_cfg.get("in_channels", 3),
            out_channels=1,
            base_channels=model_cfg.get("base_channels", 32),
            depth=model_cfg.get("depth", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        logger.info(
            "Modelo carregado de %s (época %d, Dice=%.4f)",
            self.weights_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("best_dice", 0.0),
        )
        return model, config

    def predict(
        self,
        input_path: Path | str,
        output_path: Path | str,
        report_path: Path | str | None = None,
    ) -> gpd.GeoDataFrame:
        """Executa o pipeline completo de inferência (Híbrido).

        Parameters
        ----------
        input_path: Arquivo de pontos geoespaciais de entrada.
        output_path: Caminho para o GeoJSON de saída.
        report_path: Se fornecido, salva relatório de confiança JSON.

        Returns
        -------
        GeoDataFrame em EPSG:4326 com o contrato padrão do desafio.
        """
        import main as _main  # para read_points e _working_copy

        input_path = Path(input_path)
        output_path = Path(output_path)

        # 1. Ler e validar pontos
        points = _main.read_points(input_path)
        metric_points = _main._working_copy(points)
        metric_crs = str(metric_points.crs)
        xy = np.column_stack([metric_points.geometry.x, metric_points.geometry.y])

        # 2. Rasterizar
        logger.info("Rasterizando %d pontos...", len(xy))
        raster_result = rasterize_points(xy, image_size=self.tile_size)
        tensor = torch.from_numpy(raster_result.tensor).unsqueeze(0).to(self.device)

        # 3. Inferência com o modelo
        logger.info("Executando inferência U-Net++ em %s...", self.device)
        with torch.inference_mode():
            logits = self.model(tensor)

        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
        mean_confidence = float(prob_map.mean())
        logger.info("Confiança média global da área: %.4f", mean_confidence)

        # 4. Abordagem Híbrida: Filtrar pontos pela probabilidade da U-Net++
        logger.info("Filtrando pontos através do mapa de probabilidade da IA...")
        from src.data.rasterize import world_to_pixel
        
        # Mapear coordenadas dos pontos originais para pixels no prob_map
        px_coords = world_to_pixel(xy, raster_result.transform)
        cols = np.clip(np.round(px_coords[:, 0]).astype(int), 0, prob_map.shape[1] - 1)
        rows = np.clip(np.round(px_coords[:, 1]).astype(int), 0, prob_map.shape[0] - 1)
        
        # Coletar a probabilidade para cada ponto
        point_probs = prob_map[rows, cols]
        valid_mask = point_probs >= self.threshold
        
        # Manter apenas os pontos onde a U-Net detectou uma fileira
        filtered_points = metric_points[valid_mask].copy()
        
        logger.info(
            "Filtragem concluída: %d/%d pontos mantidos (limiar=%.2f)",
            len(filtered_points), len(metric_points), self.threshold
        )
        
        if len(filtered_points) < 3:
            logger.warning("Poucos pontos sobreviveram ao filtro da U-Net++.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # 5. Pós-processamento: Gerar linhas usando o baseline geométrico determinístico
        logger.info("Acionando o motor geométrico baseline (main.py) sobre os pontos limpos...")
        from src.baseline.geometric import generate_lines_geometric
        
        # generate_lines já faz reprojeção para 4326, formatação das colunas, classificação, etc.
        # e retorna o dataframe prontinho como exigido pelo contrato
        result_wgs84 = generate_lines_geometric(filtered_points)
        
        if result_wgs84.empty:
            logger.warning("Baseline geométrico não encontrou linhas conectáveis.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # 6. Suavização (Média Móvel e 1 vértice/metro)
        logger.info("Suavizando as linhas finais (vértices a cada 1m e média móvel)...")
        from src.postprocessing.smooth import smooth_linestrings
        # Para suavizar em metros, voltamos temporariamente pro CRS UTM
        result_metric = result_wgs84.to_crs(metric_crs)
        result_metric = smooth_linestrings(result_metric, step_m=1.0, window_size=5)
        # Voltamos para o formato final exigido
        result_wgs84 = result_metric.to_crs("EPSG:4326")

        # 7. Salvar
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_wgs84.to_file(output_path, driver="GeoJSON", index=False)
        logger.info("Resultado salvo em %s (%d linhas)", output_path, len(result_wgs84))

        # 7. Relatório de confiança
        if report_path is not None:
            self._save_report(
                report_path=Path(report_path),
                n_lines=len(result_wgs84),
                mean_confidence=mean_confidence,
            )

        return result_wgs84

    @staticmethod
    def _save_report(
        report_path: Path, n_lines: int, mean_confidence: float
    ) -> None:
        """Salva relatório auxiliar de confiança em JSON."""
        report = {
            "mean_confidence": round(mean_confidence, 4),
            "detected_rows": n_lines,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Relatório de confiança salvo em %s", report_path)
