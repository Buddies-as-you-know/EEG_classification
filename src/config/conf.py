from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    type: str = "AdamW"
    params: dict = {"lr": 1e-5, "weight_decay": 0.01, "amsgrad": True}


# スケジューラーの設定
class SchedulerConfig(BaseModel):
    type: str = "ExponentialLR"
    gamma: float = 9e-1


class TrainConfig(BaseModel):
    epochs: int = 300
    batch_size: int = 512
    optimizer: OptimizerConfig = OptimizerConfig()
    loss_fn: str = "BinaryCrossEntropy"
    scheduler: SchedulerConfig = SchedulerConfig()


class ModelConfig(BaseModel):
    name: str = "EEGNet"
    n_chans: int = 32
    n_classes: int = 2
    n_times: int = 1000
    sfreq: int = 1000


class MyConfig(BaseModel):
    model_config: ModelConfig = ModelConfig()
    train_config: TrainConfig = TrainConfig()
    flip: bool = False
    data_dir: str = "./data/row/mat"
    edf_dir: str = "./data/row/edf"
    validation_name: str = "TDV"
    output_dir: str = "results/"
    wandb_projrct: str = "EEG"
    wandb_name: str = "WANDB"
