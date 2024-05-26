import torch
import pickle
import numpy as np

from src.config import ModelConfig
from src.arguments import model_args

# set device type
device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = ModelConfig(**model_args)
block_size = model_config.block_size
batch_size = model_config.batch_size

meta_path = "C:/Users/salva/Desktop/projects/eplGPT/artifacts/data/meta.pkl"


def read_meta():
    with open(meta_path, "rb") as file:
        data = pickle.load(file)

    return data


# custom data loader
def get_batch(split):
    assert split in ["train", "val"]
    data_path = f"C:/Users/salva/Desktop/projects/eplGPT/artifacts/data/{split}.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # generate a batch of data of inputs X and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    X = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + block_size + 1]).astype(np.int64)) for i in ix])

    if device == "cuda":
        # pin arrays X, y, which allows us to move them to GPU asynchronously (non_blocking=True)
        X, y = X.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        X, y = X.to(device), y.to(device)

    return X, y


meta = read_meta()
chars = meta["char_to_int"].keys()
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
encode = lambda s: [char_to_int[c] for c in s]  # encodes string to list of integers
decode = lambda l: "".join([int_to_char[i] for i in l])  # decodes list of integers to string
