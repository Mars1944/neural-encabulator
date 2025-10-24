from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


class CnnOptim:
    """
    Class-based optimizer/scheduler builder using a simple config dict.

    Usage:
      builder = CnnOptim(cfg)
      optimizer, scheduler, step_on = builder.build(model)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    @staticmethod
    def _lower(s: Any, default: str = "") -> str:
        try:
            return str(s).strip().lower()
        except Exception:
            return default

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        name = self._lower(self.cfg.get("optimizer", "adamw"), "adamw")
        lr = float(self.cfg.get("learning_rate", 1e-3))
        wd = float(self.cfg.get("weight_decay", 0.0))
        eps = float(self.cfg.get("eps", 1e-8))

        params = model.parameters()

        if name == "adam":
            betas = self.cfg.get("betas", (0.9, 0.999))
            opt = torch.optim.Adam(params, lr=lr, betas=tuple(betas), weight_decay=wd, eps=eps)
        elif name == "sgd":
            momentum = float(self.cfg.get("momentum", 0.9))
            nesterov = bool(self.cfg.get("nesterov", False))
            opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
        elif name == "rmsprop":
            momentum = float(self.cfg.get("momentum", 0.0))
            alpha = float(self.cfg.get("alpha", 0.99))
            opt = torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=wd, alpha=alpha)
        else:  # default: adamw
            betas = self.cfg.get("betas", (0.9, 0.999))
            opt = torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=wd, eps=eps)

        print(f"Optimizer: {opt.__class__.__name__} (lr={lr}, weight_decay={wd})")
        return opt

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, str]:
        name = self._lower(self.cfg.get("scheduler", "none"), "none")
        if name in ("", "none", "off", "disable"):
            return None, "epoch"

        if name == "cosine":
            t_max = int(self.cfg.get("t_max", self.cfg.get("max_epochs", 50)))
            eta_min = float(self.cfg.get("eta_min", 0.0))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            print(f"Scheduler: CosineAnnealingLR (T_max={t_max}, eta_min={eta_min}) [epoch]")
            return sch, "epoch"

        if name == "step":
            step_size = int(self.cfg.get("step_size", 30))
            gamma = float(self.cfg.get("gamma", 0.1))
            sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(f"Scheduler: StepLR (step_size={step_size}, gamma={gamma}) [epoch]")
            return sch, "epoch"

        if name == "plateau":
            patience = int(self.cfg.get("patience", 10))
            factor = float(self.cfg.get("factor", 0.5))
            min_lr = float(self.cfg.get("min_lr", 1e-6))
            cooldown = int(self.cfg.get("cooldown", 0))
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self._lower(self.cfg.get("plateau_mode", "min"), "min"),
                patience=patience,
                factor=factor,
                min_lr=min_lr,
                cooldown=cooldown,
            )
            print(
                f"Scheduler: ReduceLROnPlateau (patience={patience}, factor={factor}, min_lr={min_lr}, cooldown={cooldown}) [epoch]"
            )
            return sch, "epoch"

        if name == "warmup_cosine":
            warmup_epochs = int(self.cfg.get("warmup_epochs", 5))
            max_epochs = int(self.cfg.get("max_epochs", 50))
            warmup_epochs = max(min(warmup_epochs, max_epochs), 0)

            def warmup_lambda(current_epoch: int) -> float:
                if warmup_epochs == 0:
                    return 1.0
                return min(1.0, float(current_epoch + 1) / float(warmup_epochs))

            warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
            print(
                f"Scheduler: Warmup({warmup_epochs}) + CosineAnnealingLR (T_max={max_epochs}) [epoch]"
            )
            setattr(optimizer, "_warmup_scheduler", warm)
            setattr(optimizer, "_warmup_epochs", warmup_epochs)
            return cosine, "epoch"

        print(f"[warn] Unknown scheduler '{name}'; no scheduler will be used.")
        return None, "epoch"

    def build(
        self, model: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None, str]:
        opt = self.build_optimizer(model)
        sch, step_on = self.build_scheduler(opt)
        return opt, sch, step_on


# Backward-compatible functional API (optional)
def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:  # pragma: no cover
    return CnnOptim(cfg).build_optimizer(model)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]
) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, str]:  # pragma: no cover
    return CnnOptim(cfg).build_scheduler(optimizer)

