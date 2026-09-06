import json
import os

import torch
from torch.utils.data import Dataset


class ToutiaoDataset(Dataset):
    """今日头条文本分类数据集。

    原始数据每行使用 `_!_` 分隔字段，本项目主要用到：
    第 3 列：类别名称
    第 4 列：新闻标题文本
    """

    def __init__(self, file_path, tokenizer, label2id, max_length=128):
        # __init__ 只保存必要参数，并调用 load_data 完成数据读取。
        # 具体的读取、过滤、解析逻辑放在成员函数里，避免初始化函数太臃肿。
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        self.texts, self.labels = self.load_data()
        print(f"Loaded {len(self.texts)} samples from {self.file_path}")

    def load_data(self):
        """读取数据文件，并返回文本列表和标签列表。"""
        texts = []
        labels = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = self.parse_line(line)

                if sample is None:
                    continue

                text, label = sample
                texts.append(text)
                labels.append(label)

        return texts, labels

    def parse_line(self, line):
        """解析一行原始文本，返回 (text, label)。

        如果这一行格式不对、标题为空、类别不在 label2id 中，就返回 None。
        """
        line = line.strip()

        if not line:
            return None

        parts = line.split("_!_")

        # 至少需要：新闻 ID、类别 ID、类别名称、新闻标题。
        if len(parts) < 4:
            return None

        category_name = parts[2].strip()
        text = parts[3].strip()

        if not text:
            return None

        if category_name not in self.label2id:
            return None

        return text, self.label2id[category_name]

    def __len__(self):
        """返回数据集样本数量，DataLoader 会依赖这个方法计算 batch 数。"""
        return len(self.texts)

    def __getitem__(self, index):
        """返回一条原始样本。

        这里不做 tokenizer，也不做 padding。
        tokenizer 放到 collate_fn 中按 batch 一次性处理，更适合动态 padding。
        """
        return {
            "text": self.texts[index],
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }

    def collate_fn(self, batch):
        """把多条样本整理成模型可以直接使用的一个 batch。"""
        texts = [
            item["text"]
            for item in batch
        ]
        labels = torch.stack([
            item["labels"]
            for item in batch
        ])

        # padding="longest" 只补齐到当前 batch 中最长文本的长度。
        # truncation=True 保证超过 max_length 的文本会被截断。
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }


def get_all_labels(file_paths):
    """从数据文件中收集所有类别，并生成 label2id / id2label。"""
    labels = set()

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split("_!_")

                if len(parts) < 4:
                    continue

                category_name = parts[2].strip()

                if category_name:
                    labels.add(category_name)

    # 排序后再编号，保证每次运行生成的标签 id 顺序一致。
    labels = sorted(list(labels))

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }
    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    print("Labels:")
    for idx, label in id2label.items():
        print(f"  {idx}: {label}")

    return label2id, id2label


def load_or_create_label_map(train_path, label_map_path):
    """读取已有 label_map；如果不存在，就从训练集生成并保存。"""
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)

        id2label = {
            idx: label
            for label, idx in label2id.items()
        }
        return label2id, id2label

    label2id, id2label = get_all_labels([train_path])
    label_map_dir = os.path.dirname(label_map_path)

    if label_map_dir:
        os.makedirs(label_map_dir, exist_ok=True)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print(f"Saved label map to {label_map_path}")
    return label2id, id2label
