import argparse
import sys
import time
from pathlib import Path

import torch

from transformer import TransformerLanguageModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "high-intensity-training-the-mike-mentzer-way.txt")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "mentzer_gpt.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    def log(message):
        print(f"[{time.time() - start_time:6.1f}s] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"using device: {device}")
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(items, *args, **kwargs):
            return items

    text = args.corpus.read_text(encoding="utf-8")
    log(f"read corpus: {len(text):,} chars")
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {character: index for index, character in enumerate(chars)}
    encode = lambda value: [stoi[character] for character in value]
    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    block_size = 64
    if len(data) <= block_size + 1:
        raise ValueError("The corpus must contain more than 65 characters")
    split_index = int(0.9 * len(data))
    train_data, val_data = data[:split_index], data[split_index:]
    if len(val_data) <= block_size or len(train_data) <= block_size:
        raise ValueError("The training and validation splits must each be longer than the block size")
    log(f"encoded data: vocab={vocab_size}, train={len(train_data):,}, val={len(val_data):,}")

    batch_size = 32
    learning_rate = 5e-4
    eval_interval = 200
    n_layer, n_head, n_embd = 2, 4, 128

    def get_batch(split):
        source = train_data if split == "train" else val_data
        indices = torch.randint(len(source) - block_size, (batch_size,), device=device)
        x = torch.stack([source[index : index + block_size] for index in indices])
        y = torch.stack([source[index + 1 : index + block_size + 1] for index in indices])
        return x, y

    model = TransformerLanguageModel(
        vocab_size, block_size, n_layer, n_head, n_embd
    ).to(device)
    log(f"model built: {sum(parameter.numel() for parameter in model.parameters()):,} params")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    log(f"first forward OK, loss={loss.item():.4f}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    log("first backward OK")
    optimizer.zero_grad(set_to_none=True)

    log("starting training loop")
    for iteration in tqdm(
        range(args.max_iters), desc="training", file=sys.stdout, mininterval=0.5
    ):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                _, val_loss = model(*get_batch("val"))
            model.train()
            message = f"Iter {iteration:5d} | train {loss.item():.4f} | val {val_loss.item():.4f}"
            if hasattr(tqdm, "write"):
                tqdm.write(message)
            else:
                print(message, flush=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "chars": chars,
        "model_config": {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": 0.1,
        },
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    log(f"saved checkpoint: {args.checkpoint}")
    log("training done")


if __name__ == "__main__":
    main()
