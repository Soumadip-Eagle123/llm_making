# GPT Training on Mentzer Heavy Duty

This project trains a character-level GPT on the Mike Mentzer training corpus stored in `high-intensity-training-the-mike-mentzer-way.txt`.

## Files

- `initial.py` loads the local corpus, trains the model, and saves a checkpoint.
- `generate.py` loads a checkpoint and generates text from a prompt without retraining.
- `transformer.py` defines the Transformer language model.
- `high-intensity-training-the-mike-mentzer-way.txt` is the training corpus.

## Usage

Install PyTorch and tqdm, then train the model:

```bash
python3 initial.py --max-iters 2000
```

Training uses CUDA when available and otherwise runs on CPU. When training completes, the model is saved to `mentzer_gpt.pt`.

Run generation separately as many times as you like:

```bash
python3 generate.py --prompt "repetition" --max-new-tokens 100 --temperature 0.2 --top-k 7
```

Generation uses the saved checkpoint and does not retrain. Use `--checkpoint PATH` in either script to choose a different checkpoint location. Use `--corpus PATH` when training on another UTF-8 text corpus.
