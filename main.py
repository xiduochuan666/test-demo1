import argparse
import os
import random

import numpy as np
import swanlab
import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from dataset import ToutiaoDataset, load_or_create_label_map
from model import BertTextClassifier
from train import Trainer


def set_seed(seed):
    """固定随机种子，让实验结果尽量可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭 cudnn 的自动优化，减少同一配置多次运行时的随机波动。
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数。

    默认配置来自 JSON 文件；命令行参数只用于临时覆盖某几个配置项。
    例如：python main.py --batch-size 32 --learning-rate 3e-5
    """
    parser = argparse.ArgumentParser(
        description="Train a BERT classifier for Toutiao text classification.",
    )

    # 配置文件路径
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)

    # 数据、模型和输出路径
    parser.add_argument("--train-path", "--train_path", dest="train_path")
    parser.add_argument("--dev-path", "--dev_path", dest="dev_path")
    parser.add_argument("--test-path", "--test_path", dest="test_path")
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    parser.add_argument("--label-map-path", "--label_map_path", dest="label_map_path")

    # 常用训练超参数
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        type=float,
    )
    parser.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")

    # SwanLab 可以 online/offline/disabled。
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "offline", "disabled"],
    )
    return parser.parse_args()


def apply_arg_overrides(config, args):
    """用命令行参数覆盖 JSON 配置。

    没有传入的命令行参数保持 JSON 中的值不变。
    路径类参数会统一转成相对于 demo1 目录的绝对路径。
    """
    path_fields = [
        "train_path",
        "dev_path",
        "test_path",
        "model_dir",
        "checkpoint_dir",
        "output_dir",
        "label_map_path",
    ]
    override_fields = [
        *path_fields,
        "max_length",
        "batch_size",
        "learning_rate",
        "num_epochs",
        "dropout",
        "warmup_ratio",
        "seed",
        "device",
        "swanlab_mode",
    ]

    for field in override_fields:
        value = getattr(args, field)
        if value is not None:
            if field in path_fields:
                value = resolve_path(value)
            setattr(config, field, value)

    return config


def get_device(device_name):
    """根据配置选择训练设备。"""
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def init_swanlab(config, device):
    """初始化 SwanLab 实验记录。"""
    if config.swanlab_mode == "disabled":
        return None

    api_key = os.getenv("SWANLAB_API_KEY")

    if api_key:
        swanlab.login(
            api_key=api_key,
            save=False,
        )

    return swanlab.init(
        project=config.swanlab_project,
        workspace=config.swanlab_workspace,
        experiment_name=config.experiment_name,
        description="BERT Toutiao text classification training and evaluation",
        tags=[
            "bert-base-chinese",
            "toutiao",
            "text-classification",
        ],
        mode=config.swanlab_mode,
        config={
            **config.to_dict(),
            "device": str(device),
        },
    )


def build_dataloader(dataset, batch_size, shuffle):
    """构建 DataLoader，并使用数据集自己的 collate_fn 做动态 padding。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )


def main():
    """训练流程总入口。"""
    args = parse_args()

    # 先加载 JSON 配置，再用命令行参数覆盖。
    config = apply_arg_overrides(
        load_config(resolve_path(args.config)),
        args,
    )
    device = get_device(config.device)

    set_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 50)
    print("Device:", device)
    print("Config file:", args.config)
    print("Checkpoint directory:", config.checkpoint_dir)
    print("SwanLab mode:", config.swanlab_mode)
    print("=" * 50)

    swanlab_run = init_swanlab(config, device)

    try:
        # 标签映射必须以训练集为准，保证模型类别数量和训练标签一致。
        label2id, id2label = load_or_create_label_map(
            config.train_path,
            config.label_map_path,
        )
        num_labels = len(label2id)

        if num_labels == 0:
            raise ValueError(f"No labels found in training data: {config.train_path}")

        print("Number of labels:", num_labels)
        print("Label mapping:", label2id)

        # tokenizer 从本地 bert-base-chinese 目录加载。
        tokenizer = AutoTokenizer.from_pretrained(config.model_dir)

        # 三个数据集共用同一套 label2id，验证/测试集不会重新生成类别编号。
        train_dataset = ToutiaoDataset(
            config.train_path,
            tokenizer,
            label2id,
            config.max_length,
        )
        dev_dataset = ToutiaoDataset(
            config.dev_path,
            tokenizer,
            label2id,
            config.max_length,
        )
        test_dataset = ToutiaoDataset(
            config.test_path,
            tokenizer,
            label2id,
            config.max_length,
        )

        if len(train_dataset) == 0:
            raise ValueError(f"Training dataset is empty: {config.train_path}")
        if len(dev_dataset) == 0:
            raise ValueError(f"Dev dataset is empty: {config.dev_path}")
        if len(test_dataset) == 0:
            raise ValueError(f"Test dataset is empty: {config.test_path}")

        if swanlab_run:
            swanlab.log(
                {
                    "data/train_samples": len(train_dataset),
                    "data/dev_samples": len(dev_dataset),
                    "data/test_samples": len(test_dataset),
                    "data/num_labels": num_labels,
                },
                step=0,
            )

        train_loader = build_dataloader(
            train_dataset,
            config.batch_size,
            shuffle=True,
        )
        dev_loader = build_dataloader(
            dev_dataset,
            config.batch_size,
            shuffle=False,
        )
        test_loader = build_dataloader(
            test_dataset,
            config.batch_size,
            shuffle=False,
        )

        # num_labels 由训练集标签数量决定，避免手动配置类别数不一致。
        model = BertTextClassifier(
            model_path=config.model_dir,
            num_labels=num_labels,
            dropout=config.dropout,
        ).to(device)

        criterion = CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        # scheduler 会在训练前期 warmup，再逐步衰减学习率。
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            checkpoint_path=checkpoint_path,
        )

        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

            train_loss, train_acc = trainer.train_epoch(train_loader)
            dev_loss, dev_acc, _, _ = trainer.eval_epoch(dev_loader)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}")
            print(f"Dev Loss: {dev_loss:.4f}")
            print(f"Dev Acc: {dev_acc:.4f}")
            print(f"Best Dev Acc: {trainer.best_dev_acc:.4f}")

            if swanlab_run:
                swanlab.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "dev/loss": dev_loss,
                        "dev/accuracy": dev_acc,
                        "best/dev_accuracy": trainer.best_dev_acc,
                    },
                    step=epoch + 1,
                )

        print("\nTraining Finished!")
        print("Best Dev Accuracy:", trainer.best_dev_acc)

        # 用验证集上最好的模型做最终测试，而不是最后一个 epoch 的模型。
        print("\nLoading best model...")
        trainer.load_best_checkpoint()
        print("Best model loaded!")

        test_loss, test_acc, predictions, labels = trainer.test_epoch(test_loader)
        target_names = [
            id2label[idx]
            for idx in range(len(id2label))
        ]
        report = classification_report(
            labels,
            predictions,
            labels=list(range(len(target_names))),
            target_names=target_names,
            digits=4,
        )

        print("\n" + "=" * 50)
        print("Final Test Result")
        print("=" * 50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report")
        print(report)

        if swanlab_run:
            swanlab.log(
                {
                    "best_dev_accuracy": trainer.best_dev_acc,
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                },
                step=config.num_epochs + 1,
            )

    finally:
        # 无论训练成功还是报错，只要 SwanLab 已启动，都要正常结束 run。
        if swanlab_run:
            swanlab_run.finish()


if __name__ == "__main__":
    main()
