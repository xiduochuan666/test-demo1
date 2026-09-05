import torch
from tqdm import tqdm


def evaluate(model, dataloader, criterion, device):
    if len(dataloader) == 0:
        raise ValueError("Evaluation dataloader is empty.")

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Evaluating"
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    if total == 0:
        raise ValueError("Evaluation dataset is empty after filtering.")

    avg_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return (
        avg_loss,
        accuracy,
        all_predictions,
        all_labels
    )
