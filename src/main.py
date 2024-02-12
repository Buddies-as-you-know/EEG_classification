import hydra
from loguru import logger
from omegaconf import OmegaConf
from pydantic import ValidationError

from config.conf import MyConfig
from data.data_loader import TrainTestDasetCreate
from train_test import initialize_and_fit_clf, score_test_data, sdtdv
from untils.setup_wandb import setup_wandb


@hydra.main(
    config_name="config", version_base=None, config_path="config/Yaml/"
)
def main(cfg: MyConfig) -> None:
    setup_wandb(cfg)
    if cfg.validation_name == "SDTDV":
        sdtdv(cfg)
    else:
        for test_date in range(1, 6):
            data_class = TrainTestDasetCreate(
                cfg.data_dir,
                cfg.edf_dir,
                cfg.validation_name,
                test_date,
                sfreq=cfg.model_config.sfreq,
                flip=cfg.flip,
                denoize=False,
            )
            (
                train_set,
                val_set,
                test_data,
                test_labels,
            ) = data_class.get_train_and_test_data()
            clf = initialize_and_fit_clf(cfg, train_set, val_set)
            score_test_data(clf, test_data, test_labels, test_date, cfg)


if __name__ == "__main__":
    main()
