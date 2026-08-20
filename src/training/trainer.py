"""Loop de treinamento e validação com checkpointing, early stopping e TensorBoard.

Reprodutibilidade
-----------------
Todas as seeds (random, numpy, torch, CUDA) são configuradas via
``set_all_seeds(seed)`` antes de criar o modelo e os DataLoaders.

Checkpoint
----------
Salvo após cada época em dois arquivos:
- ``last_model.pth``: estado mais recente (para retomada)
- ``best_model.pth``: melhor Dice de validação observado

Formato do checkpoint::

    {
        "epoch": int,
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "scheduler_state_dict": ...,
        "best_dice": float,
        "config": dict,
        "metrics": dict,
    }

TensorBoard
-----------
Ativado automaticamente se tensorboard estiver instalado.
    tensorboard --logdir runs
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.training.losses import CombinedLoss
from src.training.metrics import MetricTracker, compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Configura seed para random, numpy, torch e CUDA.

    Com deterministic=True o comportamento de CUDA fica determinístico,
    mas pode ser ligeiramente mais lento. Adequado para reprodutibilidade
    de resultados científicos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orquestra o ciclo de treinamento da U-Net++.

    Parameters
    ----------
    model:
        Modelo PyTorch a treinar.
    train_loader, val_loader:
        DataLoaders de treino e validação.
    config:
        Dicionário de hiperparâmetros (lido de configs/train.yaml).
    checkpoint_dir:
        Diretório onde ``best_model.pth`` e ``last_model.pth`` serão salvos.
    log_dir:
        Diretório para logs do TensorBoard.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        checkpoint_dir: Path = Path("models"),
        log_dir: Path = Path("runs"),
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Dispositivo de treinamento: %s", self.device)
        self.model.to(self.device)

        # Hiperparâmetros de treinamento
        train_cfg = config.get("training", {})
        self.epochs: int = train_cfg.get("epochs", 100)
        self.lr: float = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay: float = train_cfg.get("weight_decay", 1e-4)
        self.patience: int = train_cfg.get("patience", 15)

        # Optimizer e scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01,
        )

        # Loss
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedLoss(
            bce_weight=loss_cfg.get("bce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            pos_weight=loss_cfg.get("pos_weight", None),
        )

        # Estado interno
        self.best_dice: float = 0.0
        self.epochs_no_improve: int = 0
        self.start_epoch: int = 0

        # TensorBoard (opcional)
        self._writer: Any = None
        self._init_tensorboard()

        # Criar diretórios
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    def _init_tensorboard(self) -> None:
        """Inicializa SummaryWriter do TensorBoard se disponível."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info("TensorBoard ativo em: %s", self.log_dir)
        except ImportError:
            logger.warning("tensorboard não instalado — logging apenas no console.")

    def train(self) -> dict[str, float]:
        """Executa o loop de treinamento completo.

        Returns
        -------
        Dicionário com as métricas finais de validação da melhor época.
        """
        logger.info(
            "Início do treinamento: %d épocas, device=%s",
            self.epochs,
            self.device,
        )

        final_metrics: dict[str, float] = {}

        for epoch in range(self.start_epoch, self.epochs):
            # Treino
            train_metrics = self._train_epoch(epoch)

            # Validação
            val_metrics = self._val_epoch(epoch)

            # Scheduler step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Checkpoint
            is_best = val_metrics.get("dice", 0.0) > self.best_dice
            if is_best:
                self.best_dice = val_metrics["dice"]
                self.epochs_no_improve = 0
                final_metrics = val_metrics
            else:
                self.epochs_no_improve += 1

            self._save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping na época %d (sem melhora por %d épocas).",
                    epoch + 1,
                    self.patience,
                )
                break

        if self._writer is not None:
            self._writer.close()

        logger.info("Treinamento concluído. Melhor Dice de validação: %.4f", self.best_dice)
        return final_metrics

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Uma época de treinamento."""
        self.model.train()
        tracker = MetricTracker()

        try:
            from tqdm import tqdm
            loader = tqdm(
                self.train_loader,
                desc=f"Treino ép.{epoch+1:02d}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
        except ImportError:
            loader = self.train_loader

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()

            # Gradient clipping para estabilidade numérica
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Métricas — usar a última saída se deep supervision
            logits_for_metrics = logits[-1] if isinstance(logits, list) else logits
            batch_metrics = compute_all_metrics(logits_for_metrics.detach(), masks)
            tracker.update(loss=loss.item(), **batch_metrics)

            # Atualizar barra de progresso com loss atual
            if hasattr(loader, 'set_postfix'):
                loader.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_metrics.get('dice', 0):.3f}")

        return tracker.compute()

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        """Uma época de validação."""
        self.model.eval()
        tracker = MetricTracker()

        try:
            from tqdm import tqdm
            loader = tqdm(
                self.val_loader,
                desc=f"  Val ép.{epoch+1:02d}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
        except ImportError:
            loader = self.val_loader

        with torch.inference_mode():
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                logits_for_metrics = logits[-1] if isinstance(logits, list) else logits
                batch_metrics = compute_all_metrics(logits_for_metrics, masks)
                tracker.update(loss=loss.item(), **batch_metrics)

                if hasattr(loader, 'set_postfix'):
                    loader.set_postfix(loss=f"{loss.item():.4f}")

        return tracker.compute()

    def _log_epoch(
        self,
        epoch: int,
        train: dict[str, float],
        val: dict[str, float],
        lr: float,
    ) -> None:
        """Loga métricas no console e no TensorBoard."""
        step = epoch + 1
        msg = (
            f"Época {step:3d} | "
            f"train_loss={train.get('loss', 0):.4f} "
            f"val_loss={val.get('loss', 0):.4f} | "
            f"val_dice={val.get('dice', 0):.4f} "
            f"val_iou={val.get('iou', 0):.4f} | "
            f"lr={lr:.2e}"
        )
        logger.info(msg)

        if self._writer is not None:
            for key, value in train.items():
                self._writer.add_scalar(f"train/{key}", value, step)
            for key, value in val.items():
                self._writer.add_scalar(f"val/{key}", value, step)
            self._writer.add_scalar("train/lr", lr, step)

    def _save_checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool
    ) -> None:
        """Salva checkpoint completo."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
            "config": self.config,
            "metrics": metrics,
        }

        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info("  → Novo melhor modelo salvo (Dice=%.4f)", self.best_dice)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Retoma treinamento a partir de um checkpoint salvo."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.start_epoch = checkpoint.get("epoch", 0)
        logger.info(
            "Checkpoint carregado: época %d, melhor Dice=%.4f",
            self.start_epoch,
            self.best_dice,
        )
