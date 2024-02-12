from omegaconf import OmegaConf

import wandb
from config.conf import MyConfig


def setup_wandb(cfg: MyConfig) -> None:
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.wandb_projrct,
        name=f"{cfg.model_config.name}_{cfg.validation_name}",
    )
