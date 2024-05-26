import torch
import argparse

from src.utils import decode
from src.config import ModelConfig
from src.model.gpt import CommentaryGPT

# set device type
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(
    prog="commentary-gpt test",
    description="tests the trained CommentaryGPT model",
    epilog="Hope you enjoy the match commentary from our artificial PETER DRURY :D"
)
parser.add_argument("--ckpt_path", nargs="?", type=str, default="C:/Users/salva/Desktop/projects/eplGPT/artifacts/checkpoint/ckpt.pt")

args = parser.parse_args()

ckpt = torch.load(args.ckpt_path, map_location=device)
n_iter = ckpt["n_iter"]
best_val_loss = ckpt["best_val_loss"]
print(f"Model trained for {n_iter} iterations with val-loss of {best_val_loss:.4f}")

model_state_dict = ckpt["model"]
model_config = ModelConfig(**ckpt["model_args"])
model = CommentaryGPT(model_config)

# adjust model_state_dict keys to match checkpoint's names
# state_dict_changes = dict(zip(model_state_dict.keys(), model.state_dict().keys()))
# for k, v in list(model_state_dict.items()):
#     new_key = state_dict_changes[k]
#     model_state_dict[new_key] = model_state_dict.pop(k)

model.to(device)
model.load_state_dict(model_state_dict)

# generate from trained checkpoint
print("Generating commentary from trained checkpoint . . . ")
prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(prompt, 500)[0].tolist()))
