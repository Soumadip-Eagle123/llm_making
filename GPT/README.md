# GPT Training on Mentzer Heavy Duty

This project is a minimal reimplementation of Andrej Karpathy's **nanoGPT** walkthrough (https://github.com/karpathy/nanoGPT). It trains a simple character-level **bigram language model** on the text of Mike Mentzer's *Heavy Duty* book provided in `human_performance_gpt_corpus.txt`.

## Files
- `initial.py` – main training script (already updated with a training loop).
- `bigramlm.py` – model definition.
- `requirements.txt` – Python dependencies (`torch`).
- `setup.sh` – quick setup script to create a virtual environment and install dependencies.

## Usage
```bash
./setup.sh           # creates venv and installs torch
source venv/bin/activate   # activate the virtual environment
python3 initial.py   # start training and generate text
```

The generated text will be printed after training completes.

## References
- Andrej Karpathy's original GPT walkthrough: https://github.com/karpathy/nanoGPT
- Mike Mentzer's *Heavy Duty* book (text supplied in this repository).
