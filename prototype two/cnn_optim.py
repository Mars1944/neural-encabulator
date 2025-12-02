from __future__ import annotations

from typing import Any, Dict, Tuple
from console import console_from_config

import torch


class CnnOptim:
    """
    Optimizer/scheduler builder using a simple config dict.

    Usage:
      builder = CnnOptim(config)
      optimizer, scheduler, step_on = builder.build(model)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def _lower(s: Any, default: str = "") -> str:
        try:
            return str(s).strip().lower()
        except Exception:
            return default

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        name = self._lower(self.config.get("optimizer", "adamw"), "adamw")
        lr = float(self.config.get("learning_rate", 1e-3))
        wd = float(self.config.get("weight_decay", 0.0))
        eps = float(self.config.get("eps", 1e-8))

        params = model.parameters()

        if name == "adam":
            betas = self.config.get("betas", (0.9, 0.999))
            optimizer = torch.optim.Adam(params, lr=lr, betas=tuple(betas), weight_decay=wd, eps=eps)
        elif name == "sgd":
            momentum = float(self.config.get("momentum", 0.9))
            nesterov = bool(self.config.get("nesterov", False))
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
        elif name == "rmsprop":
            momentum = float(self.config.get("momentum", 0.0))
            alpha = float(self.config.get("alpha", 0.99))
            optimizer = torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=wd, alpha=alpha)
        else:  # default: adamw
            betas = self.config.get("betas", (0.9, 0.999))
            optimizer = torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=wd, eps=eps)

        console_from_config(self.config).info(f"Optimizer: {optimizer.__class__.__name__} (lr={lr}, weight_decay={wd})")
        return optimizer

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, str]:
        name = self._lower(self.config.get("scheduler", "none"), "none")
        if name in ("", "none", "off", "disable"):
            return None, "epoch"

        if name == "cosine":
            t_max = int(self.config.get("t_max", self.config.get("max_epochs", 50)))
            eta_min = float(self.config.get("eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            console_from_config(self.config).info(f"Scheduler: CosineAnnealingLR (T_max={t_max}, eta_min={eta_min}) [epoch]")
            return scheduler, "epoch"

        if name == "step":
            step_size = int(self.config.get("step_size", 30))
            gamma = float(self.config.get("gamma", 0.1))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            console_from_config(self.config).info(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma}) [epoch]")
            return scheduler, "epoch"

        if name == "plateau":
            patience = int(self.config.get("patience", 10))
            factor = float(self.config.get("factor", 0.5))
            min_lr = float(self.config.get("min_lr", 1e-6))
            cooldown = int(self.config.get("cooldown", 0))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self._lower(self.config.get("plateau_mode", "min"), "min"),
                patience=patience,
                factor=factor,
                min_lr=min_lr,
                cooldown=cooldown,
            )
            console_from_config(self.config).info(
                f"Scheduler: ReduceLROnPlateau (patience={patience}, factor={factor}, min_lr={min_lr}, cooldown={cooldown}) [epoch]"
            )
            return scheduler, "epoch"

        if name == "warmup_cosine":
            warmup_epochs = int(self.config.get("warmup_epochs", 5))
            max_epochs = int(self.config.get("max_epochs", 50))
            warmup_epochs = max(min(warmup_epochs, max_epochs), 0)

            def warmup_lambda(current_epoch: int) -> float:
                if warmup_epochs == 0:
                    return 1.0
                return min(1.0, float(current_epoch + 1) / float(warmup_epochs))

            warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            console_from_config(self.config).info(
                f"Scheduler: Warmup({warmup_epochs}) + CosineAnnealingLR (T_max={max_epochs}) [epoch]"
            )
            setattr(optimizer, "_warmup_scheduler", warm)
            setattr(optimizer, "_warmup_epochs", warmup_epochs)
            return cosine, "epoch"

        console_from_config(self.config).warn(f"Unknown scheduler '{name}'; no scheduler will be used.")
        return None, "epoch"

    def build(
        self, model: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None, str]:
        optimizer = self.build_optimizer(model)
        scheduler, step_on = self.build_scheduler(optimizer)
        return optimizer, scheduler, step_on
