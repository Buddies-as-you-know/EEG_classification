from collections import defaultdict
from typing import Dict, List

import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from beartype import beartype
from braindecode import EEGClassifier
from braindecode.augmentation import AugmentedDataLoader
from braindecode.datasets import BaseConcatDataset, create_from_X_y
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from skorch.callbacks import EarlyStopping, LRScheduler, WarmRestartLR
from skorch.helper import predefined_split
from torch.nn import Module

import wandb
from config.conf import MyConfig
from data.data_loader import TrainTestDasetCreate
from data_augmentation.data_augmentation import create_transforms
from models.model import get_model


@beartype
def initialize_and_fit_clf(
    cfg: omegaconf.dictconfig.DictConfig,
    train_set: BaseConcatDataset,
    val_set: BaseConcatDataset,
) -> EEGClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{cfg.model_config.name}")
    model = get_model(
        cfg.model_config.name,
        cfg.model_config.n_chans,
        cfg.model_config.n_times,
        cfg.model_config.sfreq,
    )
    model.to(device)
    clf = EEGClassifier(
        model,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=create_transforms(),
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(val_set),
        optimizer__lr=cfg.train_config.optimizer.params["lr"],
        optimizer__weight_decay=cfg.train_config.optimizer.params[
            "weight_decay"
        ],
        batch_size=cfg.train_config.batch_size,
        callbacks=[
            "accuracy",
            (
                "early_stopping",
                EarlyStopping(monitor="valid_loss", patience=10),
            ),
            ("lr_scheduler", LRScheduler(WarmRestartLR)),
        ],
        device=device,
        classes=[0, 1],
        max_epochs=cfg.train_config.epochs,
    )
    clf.fit(train_set, y=None)
    return clf


@beartype
def sdtdv(cfg: omegaconf.dictconfig.DictConfig) -> None:
    for test_date in range(1, 6):
        for subj in range(1, 26):
            data_class = TrainTestDasetCreate(
                cfg.data_dir,
                cfg.edf_dir,
                cfg.validation_name,
                test_date,
                cfg.model_config.sfreq,
                flip=cfg.flip,
                denoize=False,
            )
            (
                train_set,
                val_set,
                test_data,
                test_labels,
            ) = data_class.get_train_and_test_data(subj=subj)
            clf = initialize_and_fit_clf(cfg, train_set, val_set)
            score_test_data(
                clf,
                test_data, 
                test_labels,
                test_date,
                cfg,
            )


@beartype
def score_test_data(
    clf: EEGClassifier,  # EEGClassifierのインスタンス
    test_data: defaultdict[
        int, np.ndarray
    ],  # テストデータ: Dict[int, np.ndarray]
    test_labels: defaultdict[
        int, List[int]
    ],  # テストラベル: Dict[int, np.ndarray]
    test_date: int,  # テスト日: int
    cfg: omegaconf.dictconfig.DictConfig,  # 設定: omegaconf.dictconfig.DictConfig
) -> None:
    logger.info("test start!")
    results_list = []  # 結果を格納するリスト

    for subj in test_data:
        test_set = create_from_X_y(
            test_data[subj],
            test_labels[subj],
            sfreq=cfg.model_config.sfreq,
            ch_names=cfg.model_config.n_chans,
            drop_last_window=True,
        )
        y_test = test_set.get_metadata().target
        pred = clf.predict(test_set)
        print(f"{set(y_test)} {pred.tolist()}")
        logger.info(f"{set(y_test)} {set(pred.tolist())}")
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred)

        # 結果をリストに追加
        results_list.append(
            [
                cfg.model_config.name,
                test_date,
                subj,
                acc,
                f1,
                roc_auc,
            ]
        )
        logger.info({cfg.model_config.name})
        logger.debug(
            f"Test acc: {(acc * 100):.2f}%, Test f1: {(f1 * 100):.2f}%"
        )

    # 結果のDataFrameを作成
    columns = [
        "model_name",
        "test_day",
        "subject",
        "Test Accuracy",
        "F1 Score",
        "ROC AUC",
    ]
    df_save = pd.DataFrame(results_list, columns=columns)
    df_save.to_csv(
        f"results/{cfg.model_config.name}_{cfg.validation_name}\
        _test_day_{test_date}.csv",
        index=False,
    )

    # WandBに記録

    wandb.log({"test_set": wandb.Table(data=results_list, columns=columns)})
