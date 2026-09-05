import os

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Trainer:
    """封装训练、验证、测试和 checkpoint 保存逻辑。

    main.py 负责组织流程，Trainer 负责真正执行一个 epoch 的训练/评估。
    这样后续想加 early stopping、混合精度、更多指标时，不用把入口文件写得太乱。
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        checkpoint_path,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path

        # 记录当前验证集最高准确率，用来判断是否保存最佳模型。
        self.best_dev_acc = 0.0

    def train_epoch(self, dataloader):
        """训练一个 epoch，并返回平均 loss 和 accuracy。"""
        self.model.train()
        total_loss = 0.0
        predictions = []
        labels = []

        for batch in tqdm(dataloader, desc="Training"):
            # 把 batch 数据移动到 CPU/GPU。
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            batch_labels = batch["labels"].to(self.device)

            # 每个 batch 训练前都要清空上一轮累积的梯度。
            self.optimizer.zero_grad()

            # 前向传播：模型输出每个类别的 logits。
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = self.criterion(logits, batch_labels)

            # 反向传播并更新参数。
            loss.backward()
            self.optimizer.step()

            # scheduler 按 step 调整学习率，通常和 optimizer.step() 配套使用。
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            batch_predictions = torch.argmax(logits, dim=1)

            # 转回 CPU list，方便 sklearn 计算 accuracy。
            predictions.extend(batch_predictions.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

        return total_loss / len(dataloader), accuracy_score(labels, predictions)

    def eval_epoch(self, dataloader):
        """在验证集上评估，并在效果更好时保存 checkpoint。"""
        loss, acc, predictions, labels = self._predict(
            dataloader,
            desc="Evaluating",
        )

        if acc > self.best_dev_acc:
            self.best_dev_acc = acc
            self.save_checkpoint()
            print("Best model saved to:", self.checkpoint_path)

        return loss, acc, predictions, labels

    def test_epoch(self, dataloader):
        """在测试集上评估，返回 loss、accuracy、预测标签和真实标签。"""
        return self._predict(
            dataloader,
            desc="Testing",
        )

    def save_checkpoint(self):
        """保存当前最佳模型，以及 optimizer/scheduler 状态。"""
        checkpoint_dir = os.path.dirname(self.checkpoint_path)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_dev_accuracy": self.best_dev_acc,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        with open(self.checkpoint_path, "wb") as f:
            torch.save(checkpoint, f)

    def load_best_checkpoint(self):
        """加载验证集上表现最好的模型权重。"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Best model not found: {self.checkpoint_path}")

        with open(self.checkpoint_path, "rb") as f:
            checkpoint = torch.load(
                f,
                map_location=self.device,
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _predict(self, dataloader, desc):
        """验证和测试共用的预测逻辑。"""
        if len(dataloader) == 0:
            raise ValueError(f"{desc} dataloader is empty.")

        self.model.eval()
        total_loss = 0.0
        predictions = []
        labels = []

        # 评估阶段不需要计算梯度，可以节省显存和加快速度。
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = self.criterion(logits, batch_labels)

                total_loss += loss.item()
                batch_predictions = torch.argmax(logits, dim=1)
                predictions.extend(batch_predictions.cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())

        if not labels:
            raise ValueError(f"{desc} dataset is empty after filtering.")

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(labels, predictions)

        return avg_loss, acc, predictions, labels
