import torch
from src.utils import decode, get_batch
from src.model.gpt import CommentaryGPT, GPTConfig

max_iters: int = 5000
eval_iters: int = 200
eval_interval: int = 500
learning_rate: float = 3e-4

config = GPTConfig()


@torch.no_grad()
def estimate_loss():
    model.eval()

    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for e in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[e] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out


model = CommentaryGPT(config)
model = model.to(config.device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"STEP {i}: TRAIN-LOSS -> {losses['train']:.4f}; VAL-LOSS -> {losses['val']:.4f}")

    # sample a batch of train data
    X, y = get_batch("train")
    # evaluate the loss
    logits, loss = model(X, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
prompt = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(prompt, max_num_tokens=500)[0].tolist()))
