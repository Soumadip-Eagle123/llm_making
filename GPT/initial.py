from pathlib import Path
import torch
from bigramlm import BigramLanguageModel
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

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
print(train_data[:block_size+1])

# for t in range(block_size):
#     context = train_data[:t+1]
#     output = train_data[t+1]
#     print(f"When input is {context}, output is {output}")

torch.manual_seed(1320)
batch_size = 4

def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print("target: ")
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        output = yb[b, t:t+1]
        print(f"If context is {decode(context.numpy(force=True))}, output is {decode(output.numpy(force=True))}")

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
