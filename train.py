"""Script de treinamento do pipeline GeoAI.

Uso
---
    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --resume models/last_model.pth

O treinamento registra logs no console e no TensorBoard:
    tensorboard --logdir runs
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
logger = logging.getLogger("train")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a U-Net++ para segmentação de fileiras geoespaciais."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Caminho para o arquivo YAML de configuração.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint para retomada de treinamento (opcional).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models"),
        help="Diretório para salvar os checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Diretório para logs do TensorBoard.",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict:
    """Carrega configuração YAML."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.error("PyYAML não instalado. Execute: pip install pyyaml")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.config.exists():
        logger.error("Arquivo de configuração não encontrado: %s", args.config)
        return 1

    config = load_config(args.config)
    logger.info("Configuração carregada de %s", args.config)

    # Importações pesadas apenas quando necessário
    from src.data.dataset import make_dataloaders
    from src.models.unetplusplus import UNetPlusPlus
    from src.training.trainer import Trainer, set_all_seeds

    # Reprodutibilidade
    seed = config.get("training", {}).get("seed", 42)
    set_all_seeds(seed)
    logger.info("Seed global configurada: %d", seed)

    # DataLoaders
    data_cfg = config.get("data", {})
    train_dl, val_dl = make_dataloaders(
        train_samples=data_cfg.get("train_samples", 10_000),
        val_samples=data_cfg.get("val_samples", 1_000),
        image_size=data_cfg.get("image_size", 256),
        batch_size=config.get("training", {}).get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 0),
        line_thickness_px=data_cfg.get("line_thickness_px", 3),
        seed=seed,
    )

    # Modelo
    model_cfg = config.get("model", {})
    model = UNetPlusPlus(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=1,
        base_channels=model_cfg.get("base_channels", 32),
        depth=model_cfg.get("depth", 4),
        deep_supervision=model_cfg.get("deep_supervision", False),
        dropout_p=model_cfg.get("dropout_p", 0.0),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Modelo: UNetPlusPlus com %s parâmetros treináveis", f"{n_params:,}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Retomar treinamento se checkpoint fornecido
    if args.resume is not None:
        if args.resume.exists():
            trainer.load_checkpoint(args.resume)
        else:
            logger.warning("Checkpoint não encontrado: %s — iniciando do zero.", args.resume)

    # Treinar
    final_metrics = trainer.train()
    logger.info("Métricas finais: %s", final_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
