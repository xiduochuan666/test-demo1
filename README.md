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
  bert-base-chinese/     # Local BERT model files, download before training
  config.py              # Config class and JSON loader
  dataset.py             # Dataset loading
  model.py               # BERT classifier
  main.py                # Training entry point
  train.py               # Trainer class
  configs/
    train_config.json    # Training paths and hyperparameters
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

The checkpoint directory and pretrained model directory are ignored by Git because these files can be large.

## Download Pretrained Model

The pretrained `bert-base-chinese` files are not stored in this repository. Download them before training and place them in:

```text
demo1/bert-base-chinese/
```

Option 1: download with Git LFS from Hugging Face:

```powershell
cd demo1
git lfs install
git clone https://huggingface.co/google-bert/bert-base-chinese bert-base-chinese
```

Option 2: download with Python:

```powershell
cd demo1
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('google-bert/bert-base-chinese').save_pretrained('bert-base-chinese'); AutoModel.from_pretrained('google-bert/bert-base-chinese').save_pretrained('bert-base-chinese')"
```

After downloading, make sure this directory contains files such as:

- `config.json`
- `vocab.txt`
- `tokenizer.json`
- `model.safetensors` or `pytorch_model.bin`

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
python main.py
```

The default training settings are stored in `configs/train_config.json`. You can edit that file for normal experiments.

You can also override the JSON settings with command-line arguments:

```powershell
python main.py --batch-size 32 --learning-rate 3e-5 --num-epochs 3 --max-length 128
```

Common arguments:

- `--config`
- `--train-path`
- `--dev-path`
- `--test-path`
- `--model-dir`
- `--checkpoint-dir`
- `--output-dir`
- `--label-map-path`
- `--max-length`
- `--batch-size`
- `--learning-rate`
- `--num-epochs`
- `--dropout`
- `--warmup-ratio`
- `--seed`
- `--device`
- `--swanlab-mode`

The script will:

1. Load labels from the training set.
2. Fine-tune `bert-base-chinese`.
3. Save the best checkpoint to `demo1/checkpoints/best_model.pt`.
4. Evaluate the best model on the test set.
5. Log metrics to SwanLab.

## Main Configuration

Edit `demo1/configs/train_config.json` to change common settings:

- `max_length`
- `batch_size`
- `learning_rate`
- `num_epochs`
- `dropout`
- `warmup_ratio`
- `device`
- `swanlab_mode`

`demo1/config.py` defines the `TrainConfig` class and loads the JSON file. The script uses CUDA automatically when `"device": "auto"` and CUDA is available, otherwise it falls back to CPU.
