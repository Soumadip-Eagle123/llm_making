import faulthandler, sys, time
faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)  # shows where it's stuck

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

log("script started, importing torch...")
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
log(f"torch {torch.__version__} imported | threads={torch.get_num_threads()} | cuda={torch.cuda.is_available()}")

try:
    from tqdm import tqdm
    log("tqdm imported OK")
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x
    log("tqdm NOT installed - progress bar disabled")

from transformer import TransformerLanguageModel
log("transformer imported")

# ---------------- Data ----------------
file_path = Path(__file__).parent / "high-intensity-training-the-mike-mentzer-way.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
log(f"read corpus: {len(text):,} chars")

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
log(f"encoded data: vocab={vocab_size}, train={len(train_data):,}, val={len(val_data):,}")

# ---------------- Hyperparameters ----------------
block_size = 64
batch_size = 32
max_iters = 2000
learning_rate = 5e-4
eval_interval = 200
n_layer, n_head, n_embd = 2, 4, 128

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# ---------------- Model ----------------
model = TransformerLanguageModel(vocab_size, block_size, n_layer, n_head, n_embd)
log(f"model built: {sum(p.numel() for p in model.parameters()):,} params")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------------- Single-step smoke test ----------------
t = time.time()
xb, yb = get_batch('train')
log("got first batch")
logits, loss = model(xb, yb)
log(f"first forward OK, loss={loss.item():.4f}, took {time.time()-t:.2f}s")
t = time.time()
loss.backward()
log(f"first backward OK, took {time.time()-t:.2f}s")

# ---------------- Training ----------------
log("starting training loop")
for it in tqdm(range(max_iters), desc="training", file=sys.stdout, mininterval=0.5):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if it % eval_interval == 0:
        with torch.no_grad():
            _, val_loss = model(*get_batch('val'))
        tqdm.write(f"Iter {it:5d} | train {loss.item():.4f} | val {val_loss.item():.4f}") if hasattr(tqdm, "write") else print(f"Iter {it} | train {loss.item():.4f} | val {val_loss.item():.4f}", flush=True)

log("training done, generating")
start = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(start, max_new_tokens=500)
print("--- Generated sample ---")
print(decode(generated[0].tolist()))
log("done")
