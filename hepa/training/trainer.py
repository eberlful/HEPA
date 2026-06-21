from datetime import datetime
import json
from re import match
import torch
import logging
import numpy as np

from pathlib import Path

import yaml
from hepa.model.hepa import HEPAModel

log = logging.getLogger(__name__)

class Trainer:

    def __init__(self, config: dict):
        self.config = config

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        # Initialize model
        self.model = HEPAModel(max_horizon=config["max_horizon"]).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        log.info(f"Optimizer {self.optimizer.__class__.__name__} initialized.")

    def _build_output_dir(self, base_dir: str, experiment_name: str) -> str | None:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            experiment_name = f"{timestamp}"

        output_path = base_path / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        return str(output_path)
    
    def _save_config(self) -> None:
        config_path = Path(self.output_dir) / "config.yaml"
        with open(config_path, "w") as f:
            config_copy = {}
            for k, v in self.config.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    config_copy[k] = v
                else:
                    config_copy[k] = str(v)
            yaml.dump(config_copy, f)

    def _save_training_history(self) -> None:
        history_path = Path(self.output_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=4)

    def save_checkpoint(
            self,
            name: str,
            phase: str,
            epoch: int,
            metrics: dict | None = None,
            optimizer_state: dict | None = None,
    ) -> Path:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics or {},
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }

        # Save optimizer state if provided
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        # checkpoint_path = self.checkpoint_dir / f"{name}_epoch{epoch}_{phase}.pt"
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> dict:
        if path == "latest":
            # Find the most recent checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in the checkpoint directory: {self.checkpoint_dir}")
            path = str(checkpoints[-1])

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore RNG states
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])

        # Restore training state
        self.current_phase = checkpoint.get("phase", "unknown")
        self.current_epoch = checkpoint.get("epoch", 0)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f" - Phase: {self.current_phase}")
        print(f" - Epoch: {self.current_epoch}")

        if checkpoint.get("metrics"):
            print(f" - Metrics: {checkpoint['metrics']}")

        return checkpoint

    def fit(self, phase: str = "all", resume_from: str | None = None) -> None:
        
        # Resume from checkpoint if specified
        resume_phase = None
        if resume_from is not None:
            checkpoint = self.load_checkpoint(resume_from)
            resumed_phase = checkpoint.get("phase", "unknown")

        # Run requested phases
        match phase:
            case "all":
                self._run_pretraining(resumed_phase)
                self._run_finetuning()
            case "pretrain":
                self._run_pretraining(resumed_phase)
            case "finetune":
                self._run_finetuning()
            case _:
                raise ValueError(f"Unknown training phase: {phase}")
            
    def _run_pretraining(self, resumed_phase: str | None = None) -> None:
        print("\n" + "=" * 60)
        print("PRETRAINING PHASE")
        print("=" * 60)

        self.current_phase = "pretrain"

        # Check if we are resuming from a checkpoint
        if resumed_phase == "pretrain":
            print(f"Resuming pretraining from checkpoint at epoch {self.current_epoch}")
            # TODO restore optimizer state and training history if needed
        elif resumed_phase == "finetune":
            print("Checkpoint is from finetuning phase, skipping pretraining and resuming finetuning.")
            return  # Skip pretraining if we are resuming from finetuning phase

        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch
            self.model.train()

            # Run one epoch
            train_metrics = self._pretrain_epoch(train_loader, alpha, grad_clip)
            val_metrics = self._pretrain_eval(val_loader, alpha)

            metrics = {
                "epoch": epoch,
                "phase": "pretrain",
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "h_std": train_metrics["h_std"],
            }

            # Handle training history
            self._save_training_history.append(metrics)
            self._save_training_history()

            # Log to callback
            if self.logger is not None:
                self.logger.log(epoch, metrics)

            print(
                f"epoch {epoch:03d} train"
            )
        