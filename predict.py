"""Script de inferência ML independente do treinamento.

Uso
---
    python predict.py \\
        --input dataset/amostra4.geojson \\
        --weights models/best_model.pth \\
        --output resultados_ml/amostra4_linhas.geojson
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferência ML: gera GeoJSON de linhas a partir de pontos usando U-Net++."
    )
    parser.add_argument("--input", type=Path, required=True, help="Arquivo de pontos de entrada.")
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint do modelo (.pth).")
    parser.add_argument("--output", type=Path, required=True, help="GeoJSON de saída.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar de binarização.")
    parser.add_argument("--tile-size", type=int, default=256, help="Tamanho do tile para inferência.")
    parser.add_argument("--report", type=Path, default=None, help="JSON de relatório de confiança (opcional).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.input.exists():
        logger.error("Entrada não encontrada: %s", args.input)
        return 1
    if not args.weights.exists():
        logger.error("Pesos do modelo não encontrados: %s", args.weights)
        return 1

    from src.inference.predictor import MLPredictor
    predictor = MLPredictor(
        weights_path=args.weights,
        threshold=args.threshold,
        tile_size=args.tile_size,
    )

    # Identificar se é arquivo único ou diretório (batch mode)
    if args.input.is_file():
        files_to_process = [args.input]
        # Se for um único arquivo e output for diretório, criar nome automático
        if args.output.is_dir() or not args.output.suffix:
            args.output.mkdir(parents=True, exist_ok=True)
            output_paths = [args.output / f"{args.input.stem}_ml.geojson"]
        else:
            output_paths = [args.output]
    else:
        # Modo Batch
        files_to_process = []
        for ext in ("*.gpkg", "*.geojson"):
            files_to_process.extend(args.input.glob(ext))
        
        # Ignorar arquivos roi e linhas corrigidas (se estiverem na mesma pasta)
        files_to_process = [f for f in files_to_process if "_roi" not in f.stem and "_linhas_corrigidas" not in f.stem]

        if not files_to_process:
            logger.error("Nenhum arquivo .gpkg ou .geojson encontrado em %s", args.input)
            return 1
            
        logger.info("Modo Lote ativado: %d arquivos encontrados.", len(files_to_process))
        args.output.mkdir(parents=True, exist_ok=True)
        output_paths = [args.output / f"{f.stem}_ml.geojson" for f in files_to_process]

    total_lines = 0
    for in_file, out_file in zip(files_to_process, output_paths):
        logger.info("Processando: %s -> %s", in_file.name, out_file.name)
        
        # Opcional: relatório para o lote (se pedido)
        rep_path = args.report.parent / f"{in_file.stem}_report.json" if args.report else None
            
        try:
            result = predictor.predict(
                input_path=in_file,
                output_path=out_file,
                report_path=rep_path,
            )
            total_lines += len(result)
            logger.info("Salvo %s com %d linhas.", out_file.name, len(result))
        except Exception as e:
            logger.error("Falha ao processar %s: %s", in_file.name, e)

    logger.info("Inferência concluída! Total de %d arquivos processados (%d linhas).", len(files_to_process), total_lines)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
