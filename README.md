# BERT Chinese Text Classification

This project trains and evaluates a Chinese text classification model based on `bert-base-chinese`.
It uses PyTorch and Hugging Face Transformers for modeling, and SwanLab for experiment tracking.

## Project Structure

```text
demo1/
  0.demo1文本分类/       # Dataset files
    train_3k.txt
    dev_1k.txt
    test_1k.txt
  bert-base-chinese/     # Local BERT tokenizer/config/model files
  config.py              # Paths and training parameters
  dataset.py             # Dataset loading
  evaluate.py            # Evaluation helpers
  model.py               # BERT classifier
  train.py               # Training entry point
  requirements.txt       # Python dependencies
```

## Requirements

- Python 3.9+
- PyTorch
- transformers
- swanlab
- scikit-learn
- tqdm
- numpy

Install dependencies:

```powershell
cd demo1
pip install -r requirements.txt
```

Install PyTorch separately if needed, following the version that matches your CUDA or CPU environment.

## Data And Model Files

The training script expects these paths:

- Dataset: `demo1/0.demo1文本分类/`
- Local BERT model: `demo1/bert-base-chinese/`
- Checkpoints: `demo1/checkpoints/`
- Outputs: `demo1/outputs/`

The checkpoint directory is ignored by Git because generated model checkpoints can be large.

## SwanLab

Training runs in SwanLab online mode. To log in without entering credentials interactively, set:

```powershell
$env:SWANLAB_API_KEY="your_api_key"
```

If this variable is not set, SwanLab will use its default login behavior.

## Train

Run training from the `demo1` directory:

```powershell
cd demo1
python train.py
```

The script will:

1. Load labels from the training set.
2. Fine-tune `bert-base-chinese`.
3. Save the best checkpoint to `demo1/checkpoints/best_model.pt`.
4. Evaluate the best model on the test set.
5. Log metrics to SwanLab.

## Main Configuration

Edit `demo1/config.py` to change common settings:

- `MAX_LENGTH`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `NUM_EPOCHS`
- `DROPOUT`

The script automatically uses CUDA when available, otherwise it falls back to CPU.
