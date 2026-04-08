"""MLP training pipeline (Section 7.2 and 8.2 of the plan).

Trains a 3-layer MLP (64->32->16) with joint classification and
reuse distance regression loss. The classification head predicts
P(optimal eviction target); the regression head predicts normalized
reuse distance as a regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset import EvictionDataset


class PageReplacementMLP(nn.Module):
    """3-layer MLP for eviction prediction (Section 7.2).

    Architecture:
        Input(num_features) -> Linear(64) -> ReLU -> Dropout
        -> Linear(32) -> ReLU -> Dropout
        -> Linear(16) -> ReLU
        -> Classification head: Linear(1) -> Sigmoid
        -> Regression head: Linear(1) -> ReLU (reuse distance)
    """

    def __init__(self, num_features: int = 27, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classify_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.reuse_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (classification_prob, reuse_distance)."""
        shared_out = self.shared(x)
        classify = self.classify_head(shared_out).squeeze(-1)
        reuse = self.reuse_head(shared_out).squeeze(-1)
        return classify, reuse

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return only classification probabilities (for inference)."""
        with torch.no_grad():
            classify, _ = self.forward(x)
        return classify


class EvictionTorchDataset(Dataset):
    """PyTorch Dataset wrapping an EvictionDataset."""

    def __init__(self, data: EvictionDataset):
        self.features = torch.from_numpy(data.features).float()
        self.is_optimal = torch.from_numpy(data.is_optimal).float()
        self.reuse_distance = torch.from_numpy(data.reuse_distance).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.is_optimal[idx], self.reuse_distance[idx]


@dataclass
class MLPTrainConfig:
    """MLP training hyperparameters (Section 7.2 table)."""
    num_features: int = 27
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    patience: int = 5
    reuse_loss_weight: float = 0.3  # Weight for auxiliary reuse distance loss
    device: str = "cpu"
    seed: int = 42


@dataclass
class MLPTrainResult:
    """Results from MLP training."""
    model: PageReplacementMLP
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


def train_mlp(
    train_data: EvictionDataset,
    val_data: EvictionDataset,
    config: MLPTrainConfig | None = None,
) -> MLPTrainResult:
    """Train the MLP model with joint classification + reuse distance loss.

    Joint loss: L = L_classify + 0.3 * L_reuse_mse

    Args:
        train_data: Training dataset with Belady-optimal labels.
        val_data: Validation dataset for early stopping.
        config: Training hyperparameters.

    Returns:
        MLPTrainResult with trained model and training history.
    """
    cfg = config or MLPTrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    model = PageReplacementMLP(
        num_features=cfg.num_features,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    classify_loss_fn = nn.BCELoss()
    reuse_loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        EvictionTorchDataset(train_data),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        EvictionTorchDataset(val_data),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for features, labels, reuse_targets in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            reuse_targets = reuse_targets.to(device)

            optimizer.zero_grad()
            classify_pred, reuse_pred = model(features)
            loss_cls = classify_loss_fn(classify_pred, labels)
            loss_reuse = reuse_loss_fn(reuse_pred, reuse_targets)
            loss = loss_cls + cfg.reuse_loss_weight * loss_reuse
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_losses.append(epoch_loss / max(num_batches, 1))

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for features, labels, reuse_targets in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                reuse_targets = reuse_targets.to(device)

                classify_pred, reuse_pred = model(features)
                loss_cls = classify_loss_fn(classify_pred, labels)
                loss_reuse = reuse_loss_fn(reuse_pred, reuse_targets)
                loss = loss_cls + cfg.reuse_loss_weight * loss_reuse
                val_loss += loss.item()
                num_val_batches += 1

        val_losses.append(val_loss / max(num_val_batches, 1))

        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return MLPTrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )


def save_model(model: PageReplacementMLP, path: str | Path) -> None:
    """Save trained MLP model."""
    torch.save(model.state_dict(), str(path))


def load_model(
    path: str | Path,
    num_features: int = 27,
) -> PageReplacementMLP:
    """Load a trained MLP model."""
    model = PageReplacementMLP(num_features=num_features)
    model.load_state_dict(torch.load(str(path), weights_only=True))
    model.eval()
    return model
