import torch.nn as nn
from transformers import BertModel


class BertTextClassifier(nn.Module):
    """基于 BERT 的文本分类模型。

    模型结构：
    BERT -> [CLS] 向量 -> Dropout -> Linear -> 分类 logits
    """

    def __init__(self, model_path, num_labels, dropout=0.1):
        super().__init__()

        # 从本地目录加载预训练 bert-base-chinese。
        self.bert = BertModel.from_pretrained(model_path)

        # BERT hidden_size 通常是 768，分类层输入维度要和它一致。
        hidden_size = self.bert.config.hidden_size

        # Dropout 可以降低过拟合风险。
        self.dropout = nn.Dropout(dropout)

        # 分类层输出维度等于类别数量。
        self.classifier = nn.Linear(
            hidden_size,
            num_labels,
        )

    def forward(self, input_ids, attention_mask):
        # attention_mask 中 1 表示真实 token，0 表示 padding token。
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # BERT 的第一个 token 是 [CLS]，常用于表示整句话的语义。
        cls_output = outputs.last_hidden_state[:, 0, :]

        cls_output = self.dropout(cls_output)

        # CrossEntropyLoss 会直接接收 logits，所以这里不需要 softmax。
        logits = self.classifier(cls_output)

        return logits
