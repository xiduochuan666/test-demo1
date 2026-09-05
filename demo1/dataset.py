import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ToutiaoDataset(Dataset):
    """
    今日头条文本分类数据集

    数据格式：
    新闻ID _!_ 分类ID _!_ 分类名称 _!_ 新闻标题 _!_ 新闻关键词
    """

    def __init__(self, file_path, tokenizer, label2id, max_length=128):
        self.texts = []
        self.labels = []

        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # 今日头条数据使用 _!_ 作为字段分隔符
                parts = line.split("_!_")

                # 至少需要：
                # ID、分类ID、分类名称、标题
                if len(parts) < 4:
                    continue

                category_name = parts[2].strip()
                text = parts[3].strip()

                # 标题为空的数据跳过
                if not text:
                    continue

                # 如果标签不在标签映射中，跳过
                if category_name not in label2id:
                    continue

                label = label2id[category_name]

                self.texts.append(text)
                self.labels.append(label)

        print(f"Loaded {len(self.texts)} samples from {file_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    input_ids = [
        item["input_ids"]
        for item in batch
    ]
    attention_masks = [
        item["attention_mask"]
        for item in batch
    ]
    labels = torch.stack([
        item["labels"]
        for item in batch
    ])

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0,
    )
    attention_masks = pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


def get_all_labels(file_paths):
    """
    从训练集等文件中读取所有类别名称，
    建立：

        label2id
        id2label

    """

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

    # 排序，保证每次运行标签编号一致
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
