import json
import os
from dataclasses import asdict, dataclass


# 当前文件所在目录，也就是 demo1 目录。
# 所有相对路径都会以这个目录为基准解析，避免从不同工作目录运行时路径错乱。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认配置文件路径。训练参数主要维护在这个 JSON 文件中。
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "train_config.json")


def resolve_path(path):
    """把配置里的相对路径转换成相对于 demo1 目录的绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


@dataclass
class TrainConfig:
    """训练配置类。

    这个类只负责描述训练需要哪些参数，不在 Python 代码里写死参数值。
    默认参数来自 JSON 配置文件，命令行参数可以在运行时临时覆盖。
    """

    # 数据集路径
    train_path: str
    dev_path: str
    test_path: str

    # 模型、输出目录和标签映射文件路径
    model_dir: str
    checkpoint_dir: str
    output_dir: str
    label_map_path: str

    # 模型和训练超参数
    max_length: int
    dropout: float
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_ratio: float

    # 运行环境和随机种子
    device: str
    seed: int

    # SwanLab 实验记录配置
    swanlab_project: str
    swanlab_workspace: str
    experiment_name: str
    swanlab_mode: str

    @classmethod
    def from_dict(cls, data):
        """把 JSON 读出来的字典转换成 TrainConfig 对象。"""
        return cls(
            train_path=resolve_path(data["train_path"]),
            dev_path=resolve_path(data["dev_path"]),
            test_path=resolve_path(data["test_path"]),
            model_dir=resolve_path(data["model_dir"]),
            checkpoint_dir=resolve_path(data["checkpoint_dir"]),
            output_dir=resolve_path(data["output_dir"]),
            label_map_path=resolve_path(data["label_map_path"]),
            max_length=int(data["max_length"]),
            dropout=float(data["dropout"]),
            batch_size=int(data["batch_size"]),
            learning_rate=float(data["learning_rate"]),
            num_epochs=int(data["num_epochs"]),
            warmup_ratio=float(data["warmup_ratio"]),
            device=data["device"],
            seed=int(data["seed"]),
            swanlab_project=data["swanlab_project"],
            swanlab_workspace=data["swanlab_workspace"],
            experiment_name=data["experiment_name"],
            swanlab_mode=data["swanlab_mode"],
        )

    def to_dict(self):
        """转换成普通字典，方便传给 SwanLab 记录完整配置。"""
        return asdict(self)


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """读取 JSON 配置文件，并返回 TrainConfig 对象。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainConfig.from_dict(data)
