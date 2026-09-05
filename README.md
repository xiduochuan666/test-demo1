# BERT Chinese Text Classification

This project fine-tunes `bert-base-chinese` for Toutiao news title classification.
The training pipeline uses PyTorch, Hugging Face Transformers, scikit-learn metrics, and SwanLab experiment logging.

## Project Structure

```text
demo1/
  0.demo1文本分类/        # Dataset files
    train_3k.txt
    dev_1k.txt
    test_1k.txt
  configs/
    train_config.json     # Default training paths and hyperparameters
  config.py               # TrainConfig class and JSON loader
  dataset.py              # Dataset, label map, and batch dynamic padding
  main.py                 # Training entry point
  model.py                # BERT classifier model
  train.py                # Trainer class
  requirements.txt        # Python dependencies
```

Generated files are not uploaded to GitHub:

- `demo1/bert-base-chinese/`
- `demo1/checkpoints/`
- `demo1/outputs/`
- `__pycache__/`
- `.vscode/`

## Install Dependencies

```powershell
cd demo1
pip install -r requirements.txt
```

If your environment already has CUDA-specific PyTorch packages, make sure the `torch` version matches your installed `torchvision` and `torchaudio`.

## Download Pretrained Model

The pretrained `bert-base-chinese` model is not stored in this repository. Download it before training and place it here:

```text
demo1/bert-base-chinese/
```

Download with Git LFS:

```powershell
cd demo1
git lfs install
git clone https://huggingface.co/google-bert/bert-base-chinese bert-base-chinese
```

Or download with Python:

```powershell
cd demo1
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('google-bert/bert-base-chinese').save_pretrained('bert-base-chinese'); AutoModel.from_pretrained('google-bert/bert-base-chinese').save_pretrained('bert-base-chinese')"
```

The directory should contain files such as:

- `config.json`
- `vocab.txt`
- `tokenizer.json`
- `model.safetensors` or `pytorch_model.bin`

## Configure Training

Default training settings are in:

```text
demo1/configs/train_config.json
```

Usually, edit this JSON file for experiments instead of changing Python code.

Important fields:

- `train_path`, `dev_path`, `test_path`
- `model_dir`
- `checkpoint_dir`
- `output_dir`
- `label_map_path`
- `max_length`
- `batch_size`
- `learning_rate`
- `num_epochs`
- `dropout`
- `warmup_ratio`
- `device`
- `swanlab_mode`

When `"device": "auto"`, the script uses CUDA if available, otherwise CPU.

## Run Training

Run from the `demo1` directory:

```powershell
cd demo1
python main.py
```

You can temporarily override JSON settings with command-line arguments:

```powershell
python main.py --learning-rate 2e-5 --batch-size 16 --dropout 0.1
```

Underscore-style aliases are also supported:

```powershell
python main.py --learning_rate 2e-5 --batch_size 16 --dropout 0.1
```

Disable SwanLab logging if needed:

```powershell
python main.py --swanlab-mode disabled
```

## Training Outputs

During training, the project generates:

```text
demo1/checkpoints/best_model.pt
demo1/outputs/label_map.json
```

`best_model.pt` stores the best model checkpoint based on validation accuracy.
`label_map.json` stores the mapping from class names to numeric label IDs.

After training, the script loads the best checkpoint and prints final test metrics, including precision, recall, f1-score, and support.
