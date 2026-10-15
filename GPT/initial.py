from pathlib import Path

file_path = Path(__file__).parent/"human_performance_gpt_corpus.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of the entire documentation is: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Hello, what can I help?"))
print(decode(encode("Hello, what can I help?")))