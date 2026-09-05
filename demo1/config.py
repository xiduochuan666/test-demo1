import os
import torch


# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# 数据路径
# =========================

DATA_DIR = os.path.join(
    BASE_DIR,
    "0.demo1文本分类"
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train_3k.txt"
)

DEV_PATH = os.path.join(
    DATA_DIR,
    "dev_1k.txt"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test_1k.txt"
)


# =========================
# BERT 模型路径
# =========================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "bert-base-chinese"
)


# =========================
# 输出目录
# =========================

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# =========================
# 模型参数
# =========================

MAX_LENGTH = 128

DROPOUT = 0.1


# =========================
# 训练参数
# =========================

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

NUM_EPOCHS = 5


# =========================
# 设备
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# 随机种子
# =========================

SEED = 42


# =========================
# 创建输出目录
# =========================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)