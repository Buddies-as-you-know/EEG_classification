from enum import Enum

import torch
import torch.nn as nn
from braindecode.models import Deep4Net, EEGConformer, EEGInceptionMI, EEGNetv4
from torcheeg.models.pyg import GIN

# GPUが使用可能かどうかを確認し、適切なデバイスを選択します。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# モデルの種類を定義する列挙体
class Model(Enum):
    EEGNet = "EEGNet"
    EEGInceptionMI = "EEGInceptionMI"
    Deep4Net = "Deep4Net"
    EEGConformer = "EEGConformer"
    GIN = "GIN"


def get_model(
    name: str, n_chans: int, n_classes: int, n_times: int = None
) -> nn.Module:
    """
    指定された名前に基づいてニューラルネットワークモデルのインスタンスを返します。

    :param name: モデルの名前
    :param n_chans: EEGチャネルの数
    :param n_classes: 分類するクラスの数
    :param n_times: サンプルの時間の長さ（必要なモデルのみ）
    :return: 初期化されたニューラルネットワークモデル
    """
    match name:
        case Model.EEGNet.name:
            model = EEGNetv4(n_chans, n_classes, input_window_samples=n_times)
        case Model.EEGInceptionMI.name:
            model = EEGInceptionMI(
                n_chans=n_chans,
                n_outputs=n_classes,
                input_window_seconds=4,
                sfreq=250,
            )
        case Model.Deep4Net.name:
            model = Deep4Net(
                n_chans=n_chans, n_classes=n_classes, n_times=n_times
            )
        case Model.EEGConformer.name:
            model = EEGConformer(
                n_chans=n_chans,
                n_outputs=n_classes,
                n_filters_time=40,
                filter_time_length=25,
                pool_time_length=75,
                pool_time_stride=15,
                drop_prob=0.5,
                att_depth=6,
                att_heads=10,
                att_drop_prob=0.5,
                final_fc_length=2440,
            )
        case Model.GIN.name:
            model = GIN(
                in_channels=n_chans, hid_channels=64, num_classes=n_classes
            )
        case _:
            raise ValueError("Unsupported model name.")
    return model.to(DEVICE)
