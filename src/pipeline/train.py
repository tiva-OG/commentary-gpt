import os
import time
import torch

from src.model.gpt import CommentaryGPT
from src.utils import decode, get_batch
from src.arguments import model_args, optim_args, train_args
from src.config import ModelConfig, OptimizerConfig, TrainConfig

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = ModelConfig(**model_args)
optim_config = OptimizerConfig(**optim_args)
train_config = TrainConfig(**train_args)


@torch.no_grad()
def estimate_loss():
    model.eval()

    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        for e in range(train_config.eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[e] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out


# BEGIN TRAINING
best_val_loss = 1e9
ckpt_path = os.path.join(train_config.output_dir, "ckpt.pt")
start_iter = 0

if not train_config.resume:
    print("Initializing new checkpoint from scratch")
    model = CommentaryGPT(model_config)
    model.to(device)

    # create output_dir if it doesn't exist
    if not os.path.exists(train_config.output_dir):
        os.makedirs(train_config.output_dir)


else:
    ckpt = torch.load(ckpt_path, map_location=device)
    start_iter = ckpt["n_iter"]
    best_val_loss = ckpt["best_val_loss"]

    print(f"Resuming training from iteration {start_iter}, with best val-loss of {best_val_loss}")
    model_state_dict = ckpt["model"]
    model_config = ModelConfig(**ckpt["model_args"])
    model = CommentaryGPT(model_config)
    model.to(device)
    model.load_state_dict(model_state_dict)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=optim_config.learning_rate)

# print number of params in checkpoint
num_model_params = sum(param.numel() for param in model.parameters()) / 1e6
print(f"Number of parameters in model: {num_model_params}M parameters")

start_time = time.time()
for n_iter in range(start_iter, train_config.max_iters):

    # evaluate loss on train & val sets after every eval_interval and write checkpoints
    if (n_iter % train_config.eval_interval == 0) or (n_iter == train_config.max_iters - 1):
        losses = estimate_loss()
        print(f"STEP {n_iter}: TRAIN-LOSS -> {losses['train']:.4f}; VAL-LOSS -> {losses['val']:.4f}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

            if n_iter > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "model_args": vars(model_config),
                    "n_iter": n_iter,
                    "best_val_loss": best_val_loss
                }
                print(f"Saving checkpoint . . . ")
                torch.save(checkpoint, ckpt_path)

    # sample batch of data
    X, y = get_batch("train")
    # evaluate loss
    logits, loss = model(X, y)
    # flush gradients asap
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    end_time = time.time()
    iter_time = end_time - start_time
    start_time = end_time

    if n_iter % train_config.log_interval == 0:
        print(f"ITERATION {n_iter}: TRAIN LOSS {loss.item():.4f}; TIME {iter_time * 1000:.2f}ms")

    n_iter += 1

# generate from trained checkpoint
# prompt = torch.tensor(encode("hello"), dtype=torch.long).unsqueeze(0)  # unsqueeze adds a new dimension at index 0
prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(prompt, 1000)[0].tolist()))
