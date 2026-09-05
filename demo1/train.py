import os
import random

import numpy as np
import swanlab
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEV_PATH,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIR,
    NUM_EPOCHS,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
)
from dataset import ToutiaoDataset, collate_fn, get_all_labels
from evaluate import evaluate
from model import BertTextClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_swanlab():
    api_key = os.getenv("SWANLAB_API_KEY")

    if api_key:
        swanlab.login(
            api_key=api_key,
            save=False,
        )

    return swanlab.init(
        project="toutiao_text_classification",
        workspace="xiduochuanhaimeng",
        experiment_name=(
            f"bert-base-chinese_lr{LEARNING_RATE}_"
            f"bs{BATCH_SIZE}_dropout{DROPOUT}"
        ),
        description="BERT Toutiao text classification training and evaluation",
        tags=[
            "bert-base-chinese",
            "toutiao",
            "text-classification",
        ],
        mode="online",
        config={
            "model": "bert-base-chinese",
            "architecture": "BertTextClassifier",
            "dataset": "toutiao-text-classification",
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "dropout": DROPOUT,
            "device": str(DEVICE),
            "seed": SEED,
        },
    )


def build_dataloader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    return total_loss / len(dataloader), correct / total


def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 50)
    print("Device:", DEVICE)
    print("Checkpoint directory:", CHECKPOINT_DIR)
    print("SwanLab mode: online")
    print("=" * 50)

    swanlab_run = init_swanlab()

    try:
        label2id, id2label = get_all_labels([TRAIN_PATH])
        num_labels = len(label2id)

        if num_labels == 0:
            raise ValueError(f"No labels found in training data: {TRAIN_PATH}")

        print("Number of labels:", num_labels)
        print("Label mapping:", label2id)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        train_dataset = ToutiaoDataset(
            TRAIN_PATH,
            tokenizer,
            label2id,
            MAX_LENGTH,
        )
        dev_dataset = ToutiaoDataset(
            DEV_PATH,
            tokenizer,
            label2id,
            MAX_LENGTH,
        )
        test_dataset = ToutiaoDataset(
            TEST_PATH,
            tokenizer,
            label2id,
            MAX_LENGTH,
        )

        if len(train_dataset) == 0:
            raise ValueError(f"Training dataset is empty: {TRAIN_PATH}")
        if len(dev_dataset) == 0:
            raise ValueError(f"Dev dataset is empty: {DEV_PATH}")
        if len(test_dataset) == 0:
            raise ValueError(f"Test dataset is empty: {TEST_PATH}")

        swanlab.log(
            {
                "data/train_samples": len(train_dataset),
                "data/dev_samples": len(dev_dataset),
                "data/test_samples": len(test_dataset),
                "data/num_labels": num_labels,
            },
            step=0,
        )

        train_loader = build_dataloader(train_dataset, shuffle=True)
        dev_loader = build_dataloader(dev_dataset, shuffle=False)
        test_loader = build_dataloader(test_dataset, shuffle=False)

        model = BertTextClassifier(
            model_path=MODEL_DIR,
            num_labels=num_labels,
            dropout=DROPOUT,
        ).to(DEVICE)

        criterion = CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        best_dev_acc = 0.0
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
            dev_loss, dev_acc, _, _ = evaluate(
                model,
                dev_loader,
                criterion,
                DEVICE,
            )

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}")
            print(f"Dev Loss: {dev_loss:.4f}")
            print(f"Dev Acc: {dev_acc:.4f}")

            swanlab.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "dev/loss": dev_loss,
                    "dev/accuracy": dev_acc,
                },
                step=epoch + 1,
            )

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc

                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                print("Saving best model to:", checkpoint_path)

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "dev_accuracy": dev_acc,
                    "epoch": epoch + 1,
                }

                with open(checkpoint_path, "wb") as f:
                    torch.save(checkpoint, f)

                swanlab.log(
                    {
                        "best/dev_accuracy": best_dev_acc,
                        "best/epoch": epoch + 1,
                    },
                    step=epoch + 1,
                )

                print("Best model saved!")

        print("\nTraining Finished!")
        print("Best Dev Accuracy:", best_dev_acc)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Best model not found: {checkpoint_path}")

        print("\nLoading best model...")
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(
                f,
                map_location=DEVICE,
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Best model loaded!")

        test_loss, test_acc, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            DEVICE,
        )

        print("\n" + "=" * 50)
        print("Final Test Result")
        print("=" * 50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        swanlab.log(
            {
                "best_dev_accuracy": best_dev_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
            },
            step=NUM_EPOCHS + 1,
        )

    finally:
        swanlab_run.finish()


if __name__ == "__main__":
    main()
