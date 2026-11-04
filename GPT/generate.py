import argparse
from pathlib import Path

import torch

from transformer import TransformerLanguageModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "mentzer_gpt.pt")
    parser.add_argument("--prompt", default="Mike mentzer")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    chars = checkpoint["chars"]
    stoi = {character: index for index, character in enumerate(chars)}
    itos = {index: character for index, character in enumerate(chars)}
    unknown_chars = sorted(set(args.prompt) - set(stoi))
    if unknown_chars:
        printable = "".join(unknown_chars)
        raise ValueError(f"Prompt contains characters not in the training corpus: {printable!r}")

    config = checkpoint["model_config"]
    model = TransformerLanguageModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_tokens = [stoi[character] for character in args.prompt]
    if not prompt_tokens:
        prompt_tokens = [0]
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    decoded = "".join(itos[index] for index in generated[0].tolist())
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    print(f"Generating with temperature={args.temperature}, top_k={args.top_k}...")
    print(decoded)


if __name__ == "__main__":
    main()
