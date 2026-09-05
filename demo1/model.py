import torch.nn as nn
from transformers import BertModel


class BertTextClassifier(nn.Module):
    """
    基于 bert-base-chinese 的文本分类模型

    结构：
        BERT
          ↓
        [CLS] 向量
          ↓
       Dropout
          ↓
       Linear
          ↓
        分类结果
    """

    def __init__(self, model_path, num_labels, dropout=0.1):
        super().__init__()

        # 加载本地 bert-base-chinese
        self.bert = BertModel.from_pretrained(model_path)

        # BERT 隐藏层维度
        hidden_size = self.bert.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 分类层
        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(self, input_ids, attention_mask):

        # BERT 前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取 [CLS] 位置的向量
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dropout
        cls_output = self.dropout(cls_output)

        # 分类
        logits = self.classifier(cls_output)

        return logits